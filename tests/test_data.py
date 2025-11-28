import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from satquant.data import DotaDataset

SAMPLE_DOTA_LINE = "10 10 20 10 20 20 10 20 plane 0"

@pytest.fixture
def mock_dataset(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    
    (images_dir / "test_img.jpg").touch()
    
    return DotaDataset(str(images_dir), str(labels_dir))

def test_parse_dota_line(mock_dataset):
    box = mock_dataset._parse_dota_line(SAMPLE_DOTA_LINE)
    assert box == (10, 10, 20, 20)

    assert mock_dataset._parse_dota_line("10 10 20 10") is None

    line_float = "10.5 10.5 20.5 10.5 20.5 20.5 10.5 20.5 plane 0"
    box = mock_dataset._parse_dota_line(line_float)
    assert box == (10, 10, 20, 20)


    line_spaces = "  10   10   20   10   20   20   10   20   plane   0  "
    box = mock_dataset._parse_dota_line(line_spaces)
    assert box == (10, 10, 20, 20)

def test_init_finds_images(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    

    
    (images_dir / "img1.jpg").touch()
    (images_dir / "img2.png").touch()
    (images_dir / "ignore.txt").touch()
    
    dataset = DotaDataset(str(images_dir), str(labels_dir))
    
    assert len(dataset.image_files) == 2
    assert any(f.endswith("img1.jpg") for f in dataset.image_files)
    assert any(f.endswith("img2.png") for f in dataset.image_files)

@patch("cv2.imread")
@patch("os.path.exists")
@patch("builtins.open", new_callable=MagicMock)
def test_crop_generator(mock_open, mock_exists, mock_imread, tmp_path):

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    
    img_path = images_dir / "test.jpg"
    img_path.touch()
    
    dataset = DotaDataset(str(images_dir), str(labels_dir))
    
    mock_exists.return_value = True

    mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_imread.return_value = mock_img
    
    mock_file = MagicMock()
    mock_file.__enter__.return_value = mock_file
    mock_file.readlines.return_value = [SAMPLE_DOTA_LINE]
    mock_open.return_value = mock_file
    
    crops = list(dataset.crop_generator())
    
    assert len(crops) == 1
    assert crops[0].shape == (640, 640, 3)

@patch("cv2.imread")
@patch("os.path.exists")
@patch("builtins.open", new_callable=MagicMock)
def test_crop_generator_skips_invalid(mock_open, mock_exists, mock_imread, tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    
    (images_dir / "img1.jpg").touch()
    (images_dir / "img2.jpg").touch()
    
    dataset = DotaDataset(str(images_dir), str(labels_dir))

    
    def side_effect_exists(path):
        if "img1.txt" in str(path): return True
        if "img2.txt" in str(path): return False
        return False
    mock_exists.side_effect = side_effect_exists
    
    mock_imread.return_value = None 
    
    mock_file = MagicMock()
    mock_file.__enter__.return_value = mock_file
    mock_file.readlines.return_value = [SAMPLE_DOTA_LINE]
    mock_open.return_value = mock_file
    
    crops = list(dataset.crop_generator())
    assert len(crops) == 0

@patch("cv2.imread")
@patch("os.path.exists")
@patch("builtins.open", new_callable=MagicMock)
def test_crop_generator_skips_metadata(mock_open, mock_exists, mock_imread, tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    
    (images_dir / "img1.jpg").touch()
    
    dataset = DotaDataset(str(images_dir), str(labels_dir))
    
    mock_exists.return_value = True
    mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    
    mock_file = MagicMock()
    mock_file.__enter__.return_value = mock_file
    mock_file.readlines.return_value = [
        "imagesource:google earth", 
        "gsd:0.1", 
        "invalid line here",
        SAMPLE_DOTA_LINE
    ]
    mock_open.return_value = mock_file
    
    crops = list(dataset.crop_generator())
    assert len(crops) == 1
