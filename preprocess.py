import os
import shutil
from PIL import Image
from rembg import remove
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- CONFIGURATION (Updated for your folder name) ---
INPUT_FOLDER = "Medicinal Leaf"   # Matches the folder in your screenshot
OUTPUT_FOLDER = "Dataset_Ready"   # This new folder will be created
TARGET_SIZE = (224, 224)          
REMOVE_BG = True                  
# ----------------------------------------------------

def process_single_image(file_info):
    src_path, dest_path = file_info
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGBA")
            
            # Optimization: Resize to 800px first to make AI fast
            img.thumbnail((800, 800), Image.LANCZOS)

            if REMOVE_BG:
                img = remove(img) # Remove background
                
                # Create White Background
                white_bg = Image.new("RGB", img.size, (255, 255, 255))
                white_bg.paste(img, mask=img.split()[3])
                img = white_bg
            else:
                img = img.convert("RGB")

            # Final Resize
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            img.save(dest_path, "JPEG", quality=95)
            return True
    except Exception as e:
        print(f"Error on {os.path.basename(src_path)}: {e}")
        return False

def main():
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    
    tasks = []
    print(f"Scanning '{INPUT_FOLDER}'...")

    for root, dirs, files in os.walk(INPUT_FOLDER):
        if '__MACOSX' in root: continue 

        for file in files:
            if file.startswith('._'): continue
            
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
                src = os.path.join(root, file)
                rel_path = os.path.relpath(root, INPUT_FOLDER)
                dest_folder = os.path.join(OUTPUT_FOLDER, rel_path)
                os.makedirs(dest_folder, exist_ok=True)
                
                dest = os.path.join(dest_folder, os.path.splitext(file)[0] + ".jpg")
                tasks.append((src, dest))

    print(f"Found {len(tasks)} images. Processing on M4 Neural Engine...")
    
    # Run in Parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(process_single_image, tasks), total=len(tasks)))

    print(f"\n✅ Done! Check the folder '{OUTPUT_FOLDER}'")

if __name__ == "__main__":
    main()