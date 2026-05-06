import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
dataset_path = "Final_Split_Data"
img_height, img_width = 224, 224
batch_size = 32 
epochs = 10
# ---------------------

def train_custom_cnn():
    # 1. SETUP DATA GENERATORS
    print("Preparing data generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode='nearest'
    )
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

    # 2. BUILD THE CUSTOM CNN (FROM SCRATCH - NO TRANSFER LEARNING)
    print("Building Custom CNN from scratch...")
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D(2, 2),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Flatten and Classify
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Crucial for a model from scratch
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    # 3. COMPILE
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 4. TRAIN
    print(f"Starting Custom CNN training for {epochs} epochs...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )

    # 5. SAVE
    model.save("custom_cnn_model.h5")
    print("✅ Model saved as 'custom_cnn_model.h5'")

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
    plt.title('Custom CNN Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Custom CNN Loss')
    plt.show()

if __name__ == "__main__":
    train_custom_cnn()