import fitz
import cv2
import numpy as np
import json
from pdf2image import convert_from_path
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

class MultimodalProcessor:
    def __init__(self, pdf_path: str, output_dir: str = "data/extracted"):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.img_dir = self.output_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.output_dir / "metadata.json"

    def process_page_images(self, page_data):
        """Worker function for threading"""
        """Convert PIL → OpenCV"""
        """Convert to grayscale"""
        """Contour detection"""
        page_num, page_image = page_data
        image_metadata = []
        open_cv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_count = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 150 and h > 150: # Slightly larger filter for 502 pages
                img_count += 1
                crop = open_cv_image[y:y+h, x:x+w]
                img_id = f"img_{page_num}_{img_count}"
                img_path = self.img_dir / f"{img_id}.png"
                cv2.imwrite(str(img_path), crop)
                image_metadata.append({
                    'image_id': img_id,
                    'page_num': page_num,
                    'path': str(img_path)
                })
        return image_metadata

    def run_full_extraction(self):
        """Checks if metadata exists, otherwise runs extraction with 4 threads."""
        if self.metadata_file.exists():
            print("Found existing metadata. Skipping extraction...")
            with open(self.metadata_file, 'r') as f:
                return json.load(f)

        print("Starting fresh extraction (using 4 threads)...")
        # 1. Extract Text
        doc = fitz.open(self.pdf_path)
        all_text = [{"text": p.get_text(), "page_num": p.number+1} for p in doc]

        # 2. Extract Images with Threading
        pages = convert_from_path(self.pdf_path, thread_count=4) # Use 4 threads for PDF rendering
        
        indexed_pages = list(enumerate(pages, 1))
        all_image_meta = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(self.process_page_images, indexed_pages))
            for res in results:
                all_image_meta.extend(res)

        # 3. Save "State" for re-use tomorrow
        combined_data = {"text": all_text, "images": all_image_meta}
        with open(self.metadata_file, 'w') as f:
            json.dump(combined_data, f)
            
        return combined_data