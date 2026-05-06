import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
dataset_path = "Final_Split_Data"
img_height, img_width = 224, 224
batch_size = 32
epochs = 10
# ---------------------

def train_vgg16():
    print("Preparing data generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_path, 'train'),
        target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical'
    )

    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(dataset_path, 'val'),
        target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical'
    )

    print("Building VGG16 model...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = False  # Freeze the base

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"Starting VGG16 training for {epochs} epochs...")
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)

    model.save("vgg16_model.h5")
    print("✅ Model saved as 'vgg16_model.h5'")

if __name__ == "__main__":
    train_vgg16()

    