import os
import cv2
import time
import numpy as np
import tensorflow as tf
import logging
from tqdm import tqdm
from typing import List, Tuple, Dict
from .data import DotaDataset

logger = logging.getLogger(__name__)

def preprocess_image(image_path: str, input_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
    """Loads, resizes, and pads image for YOLO."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    h, w = original_image.shape[:2]
    
    # Scale and Pad
    scale = min(input_size[0] / h, input_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(original_image, (new_w, new_h))
    
    canvas = np.full((input_size[0], input_size[1], 3), 128, dtype=np.uint8)
    dx = (input_size[1] - new_w) // 2
    dy = (input_size[0] - new_h) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized
    
    # Normalize
    input_data = canvas.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    return original_image, input_data, scale, dx, dy

def infer_tflite(interpreter: tf.lite.Interpreter, input_data: np.ndarray) -> np.ndarray:
    """Runs inference on the TFLite model, handling input/output quantization."""
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Handle Input Quantization
    if input_details['dtype'] != np.float32:
        scale, zero_point = input_details['quantization']
        input_data = (input_data / scale + zero_point).astype(input_details['dtype'])
        
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details['index'])
    
    # Handle Output Dequantization
    if output_details['dtype'] != np.float32:
        scale, zero_point = output_details['quantization']
        output = (output.astype(np.float32) - zero_point) * scale
        
    return output

def postprocess(output: np.ndarray, conf_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parses YOLO output to get boxes, scores, and classes."""
    # Transpose: (1, 84, 8400) -> (1, 8400, 84)
    if output.shape[1] == 84: 
        output = np.transpose(output, (0, 2, 1))
    
    boxes, scores, class_ids = [], [], []
    for i in range(output.shape[1]):
        pred = output[0][i]
        score = np.max(pred[4:])
        if score > conf_threshold:
            boxes.append(pred[:4])
            scores.append(score)
            class_ids.append(np.argmax(pred[4:]))
            
    return np.array(boxes), np.array(scores), np.array(class_ids)

def evaluate_model(model_path: str, dataset: DotaDataset, num_samples: int = 50, conf_threshold: float = 0.25) -> Dict:
    """
    Evaluates a TFLite model on a subset of the dataset.
    Returns metrics like average inference time and detection counts.
    """
    logger.info(f"[EVAL] Evaluating model: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return {}

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        logger.error(f"Failed to load TFLite model: {e}")
        return {}

    # Get input size from model
    input_details = interpreter.get_input_details()[0]
    input_shape = input_details['shape'] # [1, H, W, C]
    input_size = (input_shape[1], input_shape[2])
    logger.info(f"[EVAL] Model Input Size: {input_size}")

    samples = dataset.image_files[:num_samples]
    total_samples = len(samples)
    
    if total_samples == 0:
        logger.warning("[EVAL] No images found in dataset.")
        return {}

    total_time = 0
    total_detections = 0
    avg_conf_sum = 0
    valid_images = 0

    logger.info(f"[EVAL] Running inference on {total_samples} samples...")
    
    for img_path in tqdm(samples, desc="Evaluating"):
        try:
            _, input_data, _, _, _ = preprocess_image(img_path, input_size)
            
            start_time = time.time()
            output = infer_tflite(interpreter, input_data)
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000 # ms
            total_time += inference_time
            
            boxes, scores, _ = postprocess(output, conf_threshold)
            
            num_dets = len(scores)
            total_detections += num_dets
            if num_dets > 0:
                avg_conf_sum += np.mean(scores)
            
            valid_images += 1
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue

    if valid_images == 0:
        return {}

    avg_time = total_time / valid_images
    avg_detections = total_detections / valid_images
    avg_confidence = avg_conf_sum / valid_images if valid_images > 0 else 0 # This is avg confidence per image (of detected objects)

    metrics = {
        "samples": valid_images,
        "avg_inference_time_ms": avg_time,
        "avg_detections_per_image": avg_detections,
        "avg_confidence": avg_confidence
    }
    
    logger.info("-" * 40)
    logger.info(f"Evaluation Results ({valid_images} images):")
    logger.info(f"  Avg Inference Time: {avg_time:.2f} ms")
    logger.info(f"  Avg Detections/Img: {avg_detections:.2f}")
    logger.info(f"  Avg Confidence:     {avg_confidence:.4f}")
    logger.info("-" * 40)

    return metrics
