import os
import cv2
import numpy as np
import glob
from typing import List, Generator, Tuple


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
        # Support both jpg and png using glob character class
        search_pattern = os.path.join(self.images_dir, "*.[jp][pn]g")
        self.image_files = sorted(glob.glob(search_pattern))
        print(f"[DATA] Found {len(self.image_files)} images in {images_dir}")

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
        for img_path in self.image_files:
            # img_path is a string now
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.labels_dir, stem + ".txt")
            
            if not os.path.exists(label_path):
                print(f"[DATA] No label found for image: {img_path}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"[DATA] Failed to load image: {img_path}")
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
                    print(f"[DATA] Failed to parse DOTA line: {line}")
                    continue

                xmin, ymin, xmax, ymax = box

                # --- FOCUS CALIBRATION STRATEGY ---
                # 1. Determine object size
                obj_w = xmax - xmin
                obj_h = ymax - ymin
                
                # 2. Add context padding (padding_pct)
                # This ensures the model sees the object boundary and immediate background
                pad_w = int(obj_w * self.padding_pct)
                pad_h = int(obj_h * self.padding_pct)
                
                c_xmin = max(0, xmin - pad_w)
                c_ymin = max(0, ymin - pad_h)
                c_xmax = min(w_img, xmax + pad_w)
                c_ymax = min(h_img, ymax + pad_h)
                
                # 3. Extract the context-aware crop
                crop = img[c_ymin:c_ymax, c_xmin:c_xmax]
                
                # 4. Resize to Model Input Size (e.g., 640x640)
                if crop.size > 0:
                    crop = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
                    # Convert BGR (OpenCV default) to RGB (TensorFlow default)
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    yield crop