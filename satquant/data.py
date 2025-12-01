import logging
import os
import cv2
import numpy as np
import glob
from typing import List, Generator, Tuple
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseDataset:
    """
    Base class for datasets handling image loading and crop generation logic.
    Subclasses must implement _load_boxes.
    """
    def __init__(self, images_dir: str, labels_dir: str, crop_size: int = 640, padding_pct: float = 0.2):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.crop_size = crop_size
        self.padding_pct = padding_pct

        # Find all image files
        extensions = ["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))
        
        # Remove duplicates and sort
        self.image_files = sorted(list(set(self.image_files)))
        logger.info(f"[DATA] Found {len(self.image_files)} images in {images_dir}")

    def _load_boxes(self, label_path: str, img_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Parses the label file and returns a list of boxes in (xmin, ymin, xmax, ymax) format.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def crop_generator(self) -> Generator[np.ndarray, None, None]:
        """
        Yields focused object crops for the calibration process.
        """
        for img_path in tqdm(self.image_files, desc="Generating Crops"):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.labels_dir, stem + ".txt")
            
            if not os.path.exists(label_path):
                logger.warning(f"[DATA] No label found for image: {img_path}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"[DATA] Failed to load image: {img_path}")
                continue
            
            h_img, w_img, _ = img.shape
            
            try:
                boxes = self._load_boxes(label_path, (h_img, w_img))
            except Exception as e:
                logger.error(f"[DATA] Failed to parse label file {label_path}: {e}")
                continue

            for box in boxes:
                xmin, ymin, xmax, ymax = box
                
                # Logic extracted from original DotaDataset
                obj_w = xmax - xmin
                obj_h = ymax - ymin
                center_x = (xmin + xmax) // 2
                center_y = (ymin + ymax) // 2
                
                long_side = max(obj_w, obj_h)
                side = int(long_side * (1 + 2 * self.padding_pct))
                half_side = side // 2
                
                c_xmin = center_x - half_side
                c_ymin = center_y - half_side
                c_xmax = c_xmin + side
                c_ymax = c_ymin + side
                
                src_xmin = max(0, c_xmin)
                src_ymin = max(0, c_ymin)
                src_xmax = min(w_img, c_xmax)
                src_ymax = min(h_img, c_ymax)
                
                if src_xmax <= src_xmin or src_ymax <= src_ymin:
                    continue

                crop_img = img[src_ymin:src_ymax, src_xmin:src_xmax]
                
                pad_top = max(0, src_ymin - c_ymin)
                pad_bottom = max(0, c_ymax - src_ymax)
                pad_left = max(0, src_xmin - c_xmin)
                pad_right = max(0, c_xmax - src_xmax)
                
                if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                    crop_img = cv2.copyMakeBorder(
                        crop_img, 
                        pad_top, pad_bottom, pad_left, pad_right, 
                        cv2.BORDER_REPLICATE
                    )
                
                if crop_img.size > 0:
                    crop = cv2.resize(crop_img, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    yield crop


class DotaDataset(BaseDataset):
    """
    Handles DOTA format parsing (Oriented Bounding Boxes).
    """
    def _parse_dota_line(self, line: str) -> Tuple[int, int, int, int]:
        parts = line.strip().split()
        if len(parts) < 9:
            return None

        # Parse 8 coordinates (x1, y1 ... x4, y4)
        coords = list(map(float, parts[:8]))
        xs = coords[0::2]
        ys = coords[1::2]

        # Convert OBB to AABB (Min/Max)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        return int(xmin), int(ymin), int(xmax), int(ymax)

    def _load_boxes(self, label_path: str, img_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        boxes = []
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if "imagesource" in line or "gsd" in line:
                continue
            box = self._parse_dota_line(line)
            if box:
                boxes.append(box)
            else:
                logger.warning(f"[DATA] Failed to parse DOTA line: {line}")
        return boxes


class YoloDataset(BaseDataset):
    """
    Handles standard YOLO format parsing (class x_center y_center w h).
    Coordinates are normalized [0, 1].
    """
    def _load_boxes(self, label_path: str, img_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        h_img, w_img = img_shape
        boxes = []
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            # YOLO format: class x_center y_center w h (normalized)
            # We ignore class (parts[0]) for quantization calibration
            try:
                x_c = float(parts[1])
                y_c = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                logger.warning(f"[DATA] Invalid YOLO line: {line}")
                continue
                
            # Convert to pixels
            x_c *= w_img
            y_c *= h_img
            w *= w_img
            h *= h_img
            
            xmin = int(x_c - w / 2)
            ymin = int(y_c - h / 2)
            xmax = int(x_c + w / 2)
            ymax = int(y_c + h / 2)
            
            # Clip to image boundaries
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w_img, xmax)
            ymax = min(h_img, ymax)
            
            if xmax > xmin and ymax > ymin:
                boxes.append((xmin, ymin, xmax, ymax))
                
        return boxes