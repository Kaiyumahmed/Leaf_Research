import splitfolders
import os

# --- CONFIGURATION ---
INPUT_FOLDER = "Dataset_Ready"  # Your processed 9MB folder
OUTPUT_FOLDER = "Final_Split_Data"
# ---------------------

print(f"Splitting '{INPUT_FOLDER}' into Train/Val/Test...")

# This usually takes 5-10 seconds for small datasets
splitfolders.ratio(INPUT_FOLDER, output=OUTPUT_FOLDER,
                   seed=1337, ratio=(.7, .15, .15), group_prefix=None, move=False)

print("\n✅ Success!")
print(f"Your final research data is in: '{OUTPUT_FOLDER}'")
print("-" * 30)

# Verify the counts
for split in ['train', 'val', 'test']:
    split_path = os.path.join(OUTPUT_FOLDER, split)
    total = sum([len(files) for r, d, files in os.walk(split_path)])
    print(f"{split.upper()} set: {total} images")