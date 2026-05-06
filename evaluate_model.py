import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- CONFIGURATION ---
DATASET_PATH = "Final_Split_Data"
MODEL_PATH = "resnet_model.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# ---------------------

def evaluate():
    # 1. Load the Trained Model
    print(f"Loading {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Prepare the TEST Data (The AI has never seen this!)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    print("Loading Test Data...")
    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False # IMPORTANT: Do not shuffle, or labels will be wrong!
    )

    # 3. Run Predictions
    print("Running predictions (this takes a moment)...")
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # 4. Generate Classification Report (Precision/Recall/F1)
    print("\n" + "="*50)
    print("FINAL RESEARCH REPORT")
    print("="*50)
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

    # 5. Generate Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    evaluate()