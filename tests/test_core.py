import pytest
import tensorflow as tf
from unittest.mock import patch, MagicMock, ANY
import numpy as np
import os
from satquant.core import FocusQuantizer

@pytest.fixture
def mock_dataset():
    dataset = MagicMock()
    dataset.crop_generator.return_value = iter([np.zeros((100, 100, 3), dtype=np.uint8)])
    return dataset

@pytest.fixture
def quantizer():
    return FocusQuantizer("dummy_model_path")

def test_init():
    q = FocusQuantizer("test/path")
    assert q.model_path == "test/path"

@patch("satquant.core.tf.lite.TFLiteConverter")
@patch("os.path.isdir")
@patch("os.path.exists")
def test_convert_saved_model_full_int8(mock_exists, mock_isdir, mock_converter_cls, quantizer, mock_dataset, tmp_path):
    mock_isdir.return_value = True
    mock_exists.return_value = True 
    
    mock_converter = MagicMock()
    mock_converter_cls.from_saved_model.return_value = mock_converter
    mock_converter.convert.return_value = b"fake_tflite_model"
    
    output_path = tmp_path / "model.tflite"
    
    quantizer.convert(mock_dataset, str(output_path), mode="full_int8")
    
    mock_converter_cls.from_saved_model.assert_called_with("dummy_model_path")
    assert mock_converter.optimizations == [ANY] 
    
    assert mock_converter.target_spec.supported_ops == [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    assert mock_converter.inference_input_type == tf.int8
    assert mock_converter.inference_output_type == tf.int8
    assert mock_converter.representative_dataset is not None
    
    assert output_path.exists()
    assert output_path.read_bytes() == b"fake_tflite_model"

@patch("satquant.core.tf.lite.TFLiteConverter")
@patch("os.path.isdir")
@patch("os.path.exists")
def test_convert_keras_int16x8(mock_exists, mock_isdir, mock_converter_cls, mock_dataset, tmp_path):
    model_path = "model.h5"
    q = FocusQuantizer(model_path)
    
    mock_isdir.return_value = False
    mock_exists.return_value = True
    
    mock_converter = MagicMock()
    mock_converter_cls.from_keras_model.return_value = mock_converter
    mock_converter.convert.return_value = b"fake_tflite_model"
    
    output_path = tmp_path / "model.tflite"
    
    q.convert(mock_dataset, str(output_path), mode="int16x8")
    
    mock_converter_cls.from_keras_model.assert_called_with(model_path)
    
    assert len(mock_converter.target_spec.supported_ops) == 2
    assert mock_converter.representative_dataset is not None

@patch("satquant.core.tf.lite.TFLiteConverter")
@patch("os.path.isdir")
@patch("os.path.exists")
def test_convert_mixed(mock_exists, mock_isdir, mock_converter_cls, quantizer, mock_dataset, tmp_path):
    mock_isdir.return_value = True
    mock_exists.return_value = True
    
    mock_converter = MagicMock()
    mock_converter_cls.from_saved_model.return_value = mock_converter
    mock_converter.convert.return_value = b"fake_tflite_model"
    
    output_path = tmp_path / "model.tflite"
    
    quantizer.convert(mock_dataset, str(output_path), mode="mixed")
    
    assert len(mock_converter.target_spec.supported_ops) == 1
    assert mock_converter.target_spec.supported_ops == [ANY]

@patch("satquant.core.tf.lite.TFLiteConverter")
@patch("os.path.isdir")
@patch("os.path.exists")
def test_invalid_mode(mock_exists, mock_isdir, mock_converter_cls, quantizer, mock_dataset):
    mock_isdir.return_value = True
    mock_exists.return_value = True
    
    mock_converter = MagicMock()
    mock_converter_cls.from_saved_model.return_value = mock_converter
    
    with pytest.raises(ValueError, match="Unknown mode"):
        quantizer.convert(mock_dataset, "out.tflite", mode="invalid_mode")

@patch("os.path.isdir")
def test_model_not_found(mock_isdir, quantizer, mock_dataset):
    mock_isdir.return_value = False
    
    q = FocusQuantizer("missing.h5")
    with patch("os.path.exists", return_value=False):
        q.convert(mock_dataset, "out.tflite")

@patch("satquant.core.tf.lite.TFLiteConverter")
@patch("os.path.isdir")
@patch("os.path.exists")
def test_convert_no_normalize(mock_exists, mock_isdir, mock_converter_cls, quantizer, mock_dataset, tmp_path):
    mock_isdir.return_value = True
    mock_exists.return_value = True
    
    mock_converter = MagicMock()
    mock_converter_cls.from_saved_model.return_value = mock_converter
    mock_converter.convert.return_value = b"fake_tflite_model"
    
    output_path = tmp_path / "model.tflite"
    
    quantizer.convert(mock_dataset, str(output_path), mode="full_int8", normalize_input=False)

    assert output_path.exists()

@patch("satquant.core.tf.lite.TFLiteConverter")
@patch("os.path.isdir")
@patch("os.path.exists")
def test_convert_exception(mock_exists, mock_isdir, mock_converter_cls, quantizer, mock_dataset, tmp_path):
    mock_isdir.return_value = True
    mock_exists.return_value = True
    
    mock_converter = MagicMock()
    mock_converter_cls.from_saved_model.return_value = mock_converter
    mock_converter.convert.side_effect = Exception("Conversion failed!")
    
    output_path = tmp_path / "model.tflite"
    
    quantizer.convert(mock_dataset, str(output_path))
    
    assert not output_path.exists()
