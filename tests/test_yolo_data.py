import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from satquant.data import YoloDataset

@pytest.fixture
def mock_dataset_yolo(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    
    # Create dummy image
    (images_dir / "img1.jpg").touch()
    
    return YoloDataset(str(images_dir), str(labels_dir))

def test_yolo_parsing(mock_dataset_yolo):
    # Image size: 1000x1000
    img_shape = (1000, 1000)
    
    # YOLO Line: class x_center y_center w h (normalized)
    # Center at (500, 500), size 200x200
    # x_c=0.5, y_c=0.5, w=0.2, h=0.2
    line = "0 0.5 0.5 0.2 0.2"
    
    with patch("builtins.open", mock_open(read_data=line)):
        boxes = mock_dataset_yolo._load_boxes("dummy_path.txt", img_shape)
        
    assert len(boxes) == 1
    xmin, ymin, xmax, ymax = boxes[0]
    
    # Expected: 
    # w_px = 200, h_px = 200
    # xmin = 500 - 100 = 400
    # ymin = 500 - 100 = 400
    # xmax = 500 + 100 = 600
    # ymax = 500 + 100 = 600
    
    assert xmin == 400
    assert ymin == 400
    assert xmax == 600
    assert ymax == 600

def test_yolo_parsing_clip(mock_dataset_yolo):
    # Image size: 100x100
    img_shape = (100, 100)
    
    # Box partially out of bounds (center at 0,0)
    # x_c=0.0, y_c=0.0, w=0.2, h=0.2
    line = "0 0.0 0.0 0.2 0.2"
    
    with patch("builtins.open", mock_open(read_data=line)):
        boxes = mock_dataset_yolo._load_boxes("dummy_path.txt", img_shape)
        
    assert len(boxes) == 1
    xmin, ymin, xmax, ymax = boxes[0]
    
    # Expected:
    # w_px = 20, h_px = 20
    # raw_xmin = 0 - 10 = -10 -> clipped to 0
    # raw_ymin = 0 - 10 = -10 -> clipped to 0
    # raw_xmax = 0 + 10 = 10
    # raw_ymax = 0 + 10 = 10
    
    assert xmin == 0
    assert ymin == 0
    assert xmax == 10
    assert ymax == 10

@patch("cv2.imread")
@patch("os.path.exists")
def test_yolo_crop_generation(mock_exists, mock_imread, mock_dataset_yolo):
    mock_exists.return_value = True
    
    # Mock Image (1000x1000)
    mock_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    mock_imread.return_value = mock_img
    
    # Mock Label Content
    # Center 500,500, size 100x100
    line = "0 0.5 0.5 0.1 0.1"
    
    with patch("builtins.open", mock_open(read_data=line)):
        crops = list(mock_dataset_yolo.crop_generator())
        
    assert len(crops) == 1
    # Crop size should be equal to self.crop_size (default 640)
    assert crops[0].shape == (640, 640, 3)
