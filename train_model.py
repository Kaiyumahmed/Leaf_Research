import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
dataset_path = "Final_Split_Data"
img_height, img_width = 224, 224
batch_size = 32
epochs = 10  # How many times to loop through the data
# ---------------------

def train_brain():
    # 1. SETUP DATA GENERATORS (WITH AUGMENTATION)
    print("Preparing data generators...")
    
    # Train: Augment data to prevent overfitting
    train_datagen = ImageDataGenerator(
        rescale=1./255,          # Normalize pixel values
        rotation_range=20,       # Rotate slightly
        width_shift_range=0.2,   # Shift left/right
        height_shift_range=0.2,  # Shift up/down
        shear_range=0.2,         # Slant the image
        zoom_range=0.2,          # Zoom in/out
        horizontal_flip=True,    # Flip left/right
        fill_mode='nearest'
    )

    # Validation/Test: NO augmentation, just rescaling
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_path, 'train'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(dataset_path, 'val'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # 2. BUILD THE MODEL (Transfer Learning)
    print("Building MobileNetV2 model...")
    
    # Load the base model (pre-trained on ImageNet)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base so we don't ruin what it already knows

    # Add our custom layers on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Prevents overfitting
        layers.Dense(train_generator.num_classes, activation='softmax') # Final output layer
    ])

    # 3. COMPILE
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. TRAIN
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )

    # 5. SAVE THE MODEL
    model.save("leaf_model.h5")
    print("✅ Model saved as 'leaf_model.h5'")

    # 6. PLOT RESULTS
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

if __name__ == "__main__":
    train_brain()