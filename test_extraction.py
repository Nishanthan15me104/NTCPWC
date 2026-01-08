from src.processor import MultimodalProcessor
import os
import time

# Configuration
PDF_FILE = "Maritime_AKV_Vision_2047.pdf" 
OUTPUT_DIR = "data/extracted"

def test():
    # 1. Initialize
    processor = MultimodalProcessor(pdf_path=PDF_FILE, output_dir=OUTPUT_DIR)
    
    start_time = time.time()
    print(f"--- Starting Multimodal Extraction ---")
    print(f"Target PDF: {PDF_FILE}")
    print("Action: Text extraction + Threaded Image Cropping (4 threads)")
    
    # 2. Run the persistent/threaded process
    # This will check for metadata.json first. If not found, it runs the 4-thread process.
    data = processor.run_full_extraction()
    
    end_time = time.time()
    duration = end_time - start_time

    # 3. Verify Results
    if data:
        text_count = len(data.get("text", []))
        image_count = len(data.get("images", []))
        
        print("\n" + "="*30)
        print("✅ EXTRACTION COMPLETE")
        print(f"Total Time: {duration:.2f} seconds")
        print(f"Pages Processed (Text): {text_count}")
        print(f"Individual Images Cropped: {image_count}")
        print(f"Metadata saved at: {os.path.join(OUTPUT_DIR, 'metadata.json')}")
        print("="*30)
        
        if image_count > 0:
            print(f"Sample Image Path: {data['images'][0]['path']}")
    else:
        print("❌ Extraction failed or returned no data.")

if __name__ == "__main__":
    test()