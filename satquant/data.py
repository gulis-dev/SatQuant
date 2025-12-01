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


class DotaDataset:
    """
    Handles DOTA format parsing and 'Focus Crop' generation.
    This prepares the specific distribution of data required to minimize
    quantization Scale (S) parameter for small objects.
    """

    def __init__(self, images_dir: str, labels_dir: str, crop_size: int = 640, padding_pct: float = 0.2):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.crop_size = crop_size
        self.padding_pct = padding_pct  # Context padding to preserve object-background contrast

        # Find all image files
        extensions = ["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))
        
        # Remove duplicates and sort
        self.image_files = sorted(list(set(self.image_files)))
        logger.info(f"[DATA] Found {len(self.image_files)} images in {images_dir}")

    def _parse_dota_line(self, line: str) -> Tuple[int, int, int, int]:
        """
        Parses a single DOTA line (Oriented Bounding Box) and converts it
        to Axis-Aligned Bounding Box (AABB).
        """
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

    def crop_generator(self) -> Generator[np.ndarray, None, None]:
        """
        Yields focused object crops for the calibration process.
        Implements 'Resize Strategy' to avoid zero-padding artifacts.
        """
        for img_path in tqdm(self.image_files, desc="Generating Crops"):
            # img_path is a string now
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

            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                # Skip DOTA metadata headers
                if "imagesource" in line or "gsd" in line:
                    continue
                box = self._parse_dota_line(line)
                if box is None:
                    logger.warning(f"[DATA] Failed to parse DOTA line: {line}")
                    continue

                xmin, ymin, xmax, ymax = box

                obj_w = xmax - xmin
                obj_h = ymax - ymin
                center_x = (xmin + xmax) // 2
                center_y = (ymin + ymax) // 2
                
                # 2. Determine square size based on largest dimension + padding
                long_side = max(obj_w, obj_h)
                
                # side = long_side * (1 + 2 * self.padding_pct)
                side = int(long_side * (1 + 2 * self.padding_pct))
                half_side = side // 2
                
                # 3. Calculate crop coordinates (can be out of bounds)
                c_xmin = center_x - half_side
                c_ymin = center_y - half_side
                c_xmax = c_xmin + side
                c_ymax = c_ymin + side
                
                # 4. Handle Image Boundaries
                src_xmin = max(0, c_xmin)
                src_ymin = max(0, c_ymin)
                src_xmax = min(w_img, c_xmax)
                src_ymax = min(h_img, c_ymax)
                
                # Check if the calculated crop is valid (non-empty)
                if src_xmax <= src_xmin or src_ymax <= src_ymin:
                    continue

                crop_img = img[src_ymin:src_ymax, src_xmin:src_xmax]
                
                # 5. Pad if necessary (to maintain square shape and size)
                pad_top = src_ymin - c_ymin
                pad_bottom = c_ymax - src_ymax
                pad_left = src_xmin - c_xmin
                pad_right = c_xmax - src_xmax
                
                # Ensure non-negative
                pad_top = max(0, pad_top)
                pad_bottom = max(0, pad_bottom)
                pad_left = max(0, pad_left)
                pad_right = max(0, pad_right)
                
                if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                    crop_img = cv2.copyMakeBorder(
                        crop_img, 
                        pad_top, pad_bottom, pad_left, pad_right, 
                        cv2.BORDER_REPLICATE
                    )
                
                # 6. Resize to Model Input Size (e.g., 640x640)
                if crop_img.size > 0:
                    crop = cv2.resize(crop_img, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
                    # Convert BGR (OpenCV default) to RGB (TensorFlow default)
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    yield crop