import os
import cv2
import numpy as np
import glob
from pathlib import Path
from typing import List, Generator, Tuple


class DotaDataset:
    """
    Handles DOTA format parsing and 'Focus Crop' generation.
    This prepares the specific distribution of data required to minimize
    quantization Scale (S) parameter for small objects.
    """

    def __init__(self, images_dir: str, labels_dir: str, crop_size: int = 640, padding_pct: float = 0.2):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.crop_size = crop_size
        self.padding_pct = padding_pct  # Context padding to preserve object-background contrast

        # Find all image files
        self.image_files = sorted(list(self.images_dir.glob("*.[jp][pn]g")))
        print(f"[DATA] Found {len(self.image_files)} images in {images_dir}")

    def _parse_dota_line(self, line: str) -> Tuple[int, int, int, int]:
        """
        Parses a single DOTA line (Oriented Bounding Box) and converts it
        to Axis-Aligned Bounding Box (AABB).
        """
        parts = line.strip().split()
        if len(parts) < 9: return None  # Skip malformed lines

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
            label_path = self.labels_dir / (img_path.stem + ".txt")
            if not label_path.exists(): continue

            img = cv2.imread(str(img_path))
            if img is None: continue
            h_img, w_img, _ = img.shape

            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                # Skip DOTA metadata headers
                if "imagesource" in line or "gsd" in line: continue
                box = self._parse_dota_line(line)
                if box is None: continue

                xmin, ymin, xmax, ymax = box

                # Calculate center and crop dimensions
                cx = (xmin + xmax) // 2
                cy = (ymin + ymax) // 2

                # We define the crop size based on model input (e.g., 640x640)
                half_size = self.crop_size // 2

                # Clamp coordinates to image boundaries to prevent errors
                c_xmin = max(0, cx - half_size)
                c_ymin = max(0, cy - half_size)
                c_xmax = min(w_img, cx + half_size)
                c_ymax = min(h_img, cy + half_size)

                # Extract the crop
                crop = img[c_ymin:c_ymax, c_xmin:c_xmax]

                # --- STRATEGY: RESIZE INSTEAD OF PADDING ---
                # If the crop is smaller than target size (e.g., near edges),
                # we resize it. Zero-padding would introduce artificial black pixels (0),
                # corrupting the calibration histogram (ZeroPoint shift).
                if crop.shape[0] != self.crop_size or crop.shape[1] != self.crop_size:
                    crop = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)

                # Convert BGR (OpenCV default) to RGB (TensorFlow default)
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                yield crop