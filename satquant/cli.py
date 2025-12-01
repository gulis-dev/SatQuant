import argparse
import logging
import os
from .core import FocusQuantizer
from .data import DotaDataset, YoloDataset
from .evaluate import evaluate_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_command(args):
    logger.info("Starting Conversion Pipeline...")
    
    # 1. Initialize Dataset
    dataset_cls = YoloDataset if args.format == "yolo" else DotaDataset
    dataset = dataset_cls(
        images_dir=os.path.join(args.data, "images"),
        labels_dir=os.path.join(args.data, "labelTxt") if args.format == "dota" else os.path.join(args.data, "labels"),
        crop_size=args.crop_size,
        padding_pct=args.padding_pct
    )
    
    # 2. Initialize Quantizer
    quantizer = FocusQuantizer(model_path=args.model)
    
    # 3. Convert
    quantizer.convert(
        dataset=dataset,
        output_path=args.output,
        mode=args.mode,
        normalize_input=not args.no_normalize
    )

def evaluate_command(args):
    logger.info("Starting Evaluation Pipeline...")
    
    # 1. Initialize Dataset (for evaluation images)
    dataset_cls = YoloDataset if args.format == "yolo" else DotaDataset
    dataset = dataset_cls(
        images_dir=os.path.join(args.data, "images"),
        labels_dir=os.path.join(args.data, "labelTxt") if args.format == "dota" else os.path.join(args.data, "labels"),
        crop_size=args.crop_size # Not used for full image eval but required by init
    )
    
    evaluate_model(
        model_path=args.model,
        dataset=dataset,
        num_samples=args.num_samples,
        conf_threshold=args.conf_threshold
    )

def main():
    parser = argparse.ArgumentParser(description="SatQuant: Satellite Imagery Quantization Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Convert Command
    parser_convert = subparsers.add_parser("convert", help="Quantize a model using Focus Calibration")
    parser_convert.add_argument("--model", required=True, help="Path to input model (SavedModel dir or .h5)")
    parser_convert.add_argument("--data", required=True, help="Path to dataset root")
    parser_convert.add_argument("--format", default="dota", choices=["dota", "yolo"], help="Dataset format (dota: OBB, yolo: standard)")
    parser_convert.add_argument("--output", required=True, help="Path to save quantized TFLite model")
    parser_convert.add_argument("--mode", default="full_int8", choices=["full_int8", "int16x8", "mixed"], help="Quantization mode")
    parser_convert.add_argument("--crop_size", type=int, default=640, help="Calibration crop size")
    parser_convert.add_argument("--padding_pct", type=float, default=0.2, help="Context padding percentage")
    parser_convert.add_argument("--no-normalize", action="store_true", help="Disable input normalization (0-1)")
    parser_convert.set_defaults(func=convert_command)
    
    # Evaluate Command
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate a quantized model")
    parser_eval.add_argument("--model", required=True, help="Path to TFLite model")
    parser_eval.add_argument("--data", required=True, help="Path to dataset root")
    parser_eval.add_argument("--format", default="dota", choices=["dota", "yolo"], help="Dataset format (dota: OBB, yolo: standard)")
    parser_eval.add_argument("--num_samples", type=int, default=50, help="Number of images to evaluate")
    parser_eval.add_argument("--conf_threshold", type=float, default=0.25, help="Confidence threshold")
    parser_eval.add_argument("--crop_size", type=int, default=640, help="Dummy arg for dataset init")
    parser_eval.set_defaults(func=evaluate_command)
    
    args = parser.parse_args()
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
