import pytest
import numpy as np
import sys
from unittest.mock import patch, MagicMock
from satquant.evaluate import evaluate_model
from satquant.cli import main

# --- Tests for evaluate.py ---

@pytest.fixture
def mock_dataset_eval():
    dataset = MagicMock()
    dataset.image_files = ["img1.jpg", "img2.jpg"]
    return dataset

@patch("satquant.evaluate.tf.lite.Interpreter")
@patch("satquant.evaluate.cv2.imread")
@patch("os.path.exists")
def test_evaluate_model_success(mock_exists, mock_imread, mock_interpreter_cls, mock_dataset_eval):
    # Setup mocks
    mock_exists.return_value = True
    
    # Mock Image
    mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Mock Interpreter
    mock_interpreter = MagicMock()
    mock_interpreter_cls.return_value = mock_interpreter
    
    # Mock Input/Output Details
    mock_interpreter.get_input_details.return_value = [{'index': 0, 'shape': [1, 640, 640, 3], 'dtype': np.float32}]
    mock_interpreter.get_output_details.return_value = [{'index': 1, 'dtype': np.float32}]
    
    dummy_output = np.zeros((1, 84, 10), dtype=np.float32)

    dummy_output[0, 4, 0] = 0.9 # Score > 0.25
    
    mock_interpreter.get_tensor.return_value = dummy_output
    
    # Run evaluation
    metrics = evaluate_model("dummy_model.tflite", mock_dataset_eval, num_samples=2)
    
    assert metrics["samples"] == 2
    assert metrics["avg_detections_per_image"] == 1.0
    assert metrics["avg_confidence"] == 0.9
    assert "avg_inference_time_ms" in metrics

@patch("satquant.evaluate.tf.lite.Interpreter")
@patch("os.path.exists")
def test_evaluate_model_no_samples(mock_exists, mock_interpreter_cls, mock_dataset_eval):
    mock_exists.return_value = True
    mock_dataset_eval.image_files = [] # Empty dataset
    
    metrics = evaluate_model("dummy_model.tflite", mock_dataset_eval)
    
    assert metrics == {}


# --- Tests for cli.py ---

@patch("satquant.cli.FocusQuantizer")
@patch("satquant.cli.DotaDataset")
def test_cli_convert(mock_dataset_cls, mock_quantizer_cls):
    test_args = [
        "satquant", "convert",
        "--model", "model_dir",
        "--data", "data_dir",
        "--output", "out.tflite",
        "--mode", "int16x8"
    ]
    
    with patch.object(sys, 'argv', test_args):
        main()
        
    mock_dataset_cls.assert_called_once()
    mock_quantizer_cls.assert_called_with(model_path="model_dir")
    mock_quantizer_cls.return_value.convert.assert_called_once()
    
    # Check arguments passed to convert
    _, kwargs = mock_quantizer_cls.return_value.convert.call_args
    assert kwargs["output_path"] == "out.tflite"
    assert kwargs["mode"] == "int16x8"

@patch("satquant.cli.evaluate_model")
@patch("satquant.cli.DotaDataset")
def test_cli_evaluate(mock_dataset_cls, mock_evaluate):
    test_args = [
        "satquant", "evaluate",
        "--model", "model.tflite",
        "--data", "data_dir",
        "--num_samples", "10"
    ]
    
    with patch.object(sys, 'argv', test_args):
        main()
        
    mock_dataset_cls.assert_called_once()
    mock_evaluate.assert_called_once()
    
    # Check arguments passed to evaluate
    _, kwargs = mock_evaluate.call_args
    assert kwargs["model_path"] == "model.tflite"
    assert kwargs["num_samples"] == 10
