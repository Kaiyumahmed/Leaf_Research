import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# --- CONFIGURATION ---
# Point this to ONE specific image in your train folder
IMAGE_PATH = "Final_Split_Data/train/Neem/Neem_1.jpg" # <--- CHANGE THIS to a real file path
OUTPUT_FOLDER = "Augmentation_Samples"
# ---------------------

def visualize():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 1. Setup the Generator (Same settings as your training)
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # 2. Load one image
    img = load_img(IMAGE_PATH, target_size=(224, 224))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape) # Reshape to (1, 224, 224, 3)

    # 3. Generate 5 variations and save them
    print("Generating 5 augmented examples...")
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=OUTPUT_FOLDER, save_prefix='aug', save_format='jpeg'):
        i += 1
        if i >= 5:
            break  # Stop after 5 images

    print(f"✅ Saved 5 examples to '{OUTPUT_FOLDER}'. Use these in your paper!")

if __name__ == "__main__":
    visualize()