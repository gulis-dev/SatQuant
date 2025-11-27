from satquant import FocusQuantizer, DotaDataset

# CONFIG
IMAGES_DIR = "./dota_samples/images"
LABELS_DIR = "./dota_samples/labelTxt"
MODEL_PATH = "./yolov8n_saved_model"

# 1. Preparing DOTA Dataset z Focus Crops
dataset = DotaDataset(
    images_dir=IMAGES_DIR, 
    labels_dir=LABELS_DIR, 
    crop_size=640,
    padding_pct=0.2
)

# 3. Quantizer Initialization
quantizer = FocusQuantizer(model_path=MODEL_PATH)

# 4. Conversion to TFLite with Full INT8
quantizer.convert(
    dataset=dataset,
    output_path="yolov8_satellite_quantized.tflite",
    mode="int16x8", 
    normalize_input=True 
)

print("Done!")