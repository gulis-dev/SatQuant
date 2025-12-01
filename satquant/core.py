import tensorflow as tf
import numpy as np
import os
import logging
from .data import DotaDataset

logger = logging.getLogger(__name__)


class FocusQuantizer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        logger.info(f"[CORE] Initialized Quantizer for: {model_path}")

    def convert(self, dataset: DotaDataset, output_path: str, mode: str = "full_int8", normalize_input: bool = True):
        """
        Executes the TFLite conversion pipeline with custom calibration.
        """

        # Generator required by TFLite API to feed calibration data
        def _tf_gen():
            for crop in dataset.crop_generator():
                # Safety check: skip empty or corrupted crops to prevent crash
                if crop is None or crop.size == 0:
                    continue

                crop = crop.astype(np.float32)

                # Normalize pixel values to [0, 1] range if expected by the model architecture
                if normalize_input:
                    crop = crop / 255.0

                # Add batch dimension (H, W, C) -> (1, H, W, C) as required by TFLite
                crop = np.expand_dims(crop, axis=0)
                yield [crop]

        logger.info(f"[CORE] Loading model from: {self.model_path}")

        converter = None

        # 1. Detect model format & Load
        if os.path.isdir(self.model_path):
            # Check for standard TensorFlow SavedModel structure
            if not os.path.exists(os.path.join(self.model_path, "saved_model.pb")):
                logger.error(f"[ERROR] Path is a directory but 'saved_model.pb' is missing.")
                return
            converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)

        # Support for legacy Keras formats (.h5, .keras)
        elif self.model_path.endswith(".h5") or self.model_path.endswith(".keras"):
            if not os.path.exists(self.model_path):
                logger.error(f"[ERROR] Model file not found: {self.model_path}")
                return
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model_path)

        else:
            logger.error(f"[ERROR] Unknown model format. Please provide a SavedModel directory or .h5/.keras file.")
            return

        # 2. Configure Optimization Strategy
        # Enable default optimizations (quantization)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Inject our custom Focus Calibration dataset
        # converter.representative_dataset = _tf_gen # This line is moved into the full_int8 block

        # 3. Configure Hardware Compatibility
        if mode == "full_int8":
            logger.info("[CORE] Mode: FULL INT8 (Hardware Safe - Edge TPU/NPU)")
            # Enforce integer-only operations (critical for Edge TPU)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # Enforce integer input/output interface to avoid runtime casting
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            # Calibration is required for full_int8
            converter.representative_dataset = _tf_gen

        elif mode == "int16x8":
            logger.info("[CORE] Mode: INT16x8 (High Precision - Best for YOLO/Regression)")
            # Weights are INT8 (small size), Activations are INT16 (high precision)
            # This fixes the "0 mAP" issue on YOLO models by providing 65k quantization levels for regression heads.
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
            # Note: Edge TPU might not support INT16 activations fully, falling back to CPU for those ops.
            converter.representative_dataset = _tf_gen

        elif mode == "mixed":
            logger.info("[CORE] Mode: MIXED PRECISION (Dynamic Range - Weights INT8, Activations Float)")
            # Dynamic Range Quantization (No representative dataset needed)
            # This preserves float32 activations, avoiding the fused-head quantization issue.
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            
        else:
            raise ValueError("Unknown mode! Use 'full_int8', 'int16x8', or 'mixed'")

        # 4. Run Conversion
        try:
            logger.info("[CORE] Starting conversion (this may take a while)...")
            tflite_model = converter.convert()

            with open(output_path, 'wb') as f:
                f.write(tflite_model)

            with open(output_path, 'wb') as f:
                f.write(tflite_model)

            logger.info(f"[SUCCESS] Model saved to: {output_path}")
            logger.info(f"[INFO] Size: {len(tflite_model) / 1024 / 1024:.2f} MB")

        except Exception as e:
            logger.error(f"\n[ERROR] Conversion failed!")
            logger.error(f"Details: {str(e)}")
            if "mass mismatch" in str(e) or "shape" in str(e):
                logger.warning("[HINT] Check if dataset crop_size matches model input shape.")