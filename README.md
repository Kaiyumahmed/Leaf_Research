# Medicinal Plant Leaf Classification: Model Training 🧪

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

This repository contains the machine learning pipeline, data preprocessing scripts, and model training notebooks for the research study *"A Machine Learning Approach to Identify and Classify Medicinal Plant Leaves."* The ultimate output of this repository is a highly optimized, 14MB TensorFlow Lite (`.tflite`) model deployed natively on edge devices via the [MediDitect Flutter App](#).

---

## 📊 Dataset Details

The model is trained on a proprietary, real-world dataset compiled specifically for this research. The images were captured via smartphone cameras under varying lighting conditions, backgrounds, and angles to ensure high generalization for edge deployment.

* **Total Images:** 1,750
* **Classes:** 5 (350 images per class)
* **Target Species:**
    1. Indian Pennywort (Thankuni)
    2. Henna (Mehedi)
    3. False Daisy (Kalo Keshi)
    4. Basil (Tulsi)
    5. Neem

---

## 🧠 Model Architecture & Evaluation

To determine the optimal architecture for mobile deployment, five distinct Convolutional Neural Networks (CNNs) were evaluated against the dataset:
1. Custom CNN
2. VGG16
3. ResNet50
4. DenseNet121
5. **MobileNetV2** (Selected)

**Why MobileNetV2?**
While larger models performed well, MobileNetV2 (utilizing Depthwise Separable Convolutions) provided the perfect intersection of diagnostic accuracy and computational efficiency.
* **Final Validation Accuracy:** 98.5%
* **Parameters:** ~2.2 Million
* **Exported TFLite Size:** ~14 MB

---

## ⚙️ Training Pipeline

The training workflow is structured as follows:

1.  **Data Preprocessing:** * Images resized to `224x224` pixels.
    * RGB channels normalized to a `[0, 1]` scale.
2.  **Data Augmentation:**
    * Applied rotation, zoom, horizontal/vertical flipping, and contrast shifts to artificially expand the dataset and prevent overfitting on specific lighting conditions.
3.  **Transfer Learning:**
    * Loaded MobileNetV2 with pre-trained ImageNet weights.
    * Froze base layers and attached a custom fully connected head with Softmax activation.
4.  **Optimization:**
    * Categorical Crossentropy loss function.
    * Adam Optimizer with dynamic learning rate scheduling.
5.  **Conversion:**
    * Exported the standard `.h5` / `SavedModel` to `.tflite` format for mobile deployment.

---

## 📂 Repository Structure

```text
├── dataset/                   # Raw and augmented image data (Excluded from git tracking)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation_and_tflite_conversion.ipynb
├── models/                    # Saved .h5 and .tflite outputs
├── requirements.txt           # Python environment dependencies
└── README.md

```

---

## 🚀 How to Run Locally

### 1. Setup Environment

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Prepare Data

Place your dataset in the `dataset/` directory, organized by class folders (e.g., `dataset/Neem/`, `dataset/Basil/`).

### 3. Run Jupyter Notebooks

Launch Jupyter to explore the training process step-by-step:

```bash
jupyter notebook

```

Execute the notebooks in the `notebooks/` directory sequentially. The final output will be a `leaf_model.tflite` file generated in the `models/` directory.

---

## 👥 Authors

* **Kaiyum Ahmed** (ID: 222-15-6***)
* **S.M. Mojahedul Islam** (ID: 222-15-6***)

**Department of Computer Science and Engineering (CSE)**
*Daffodil International University*

```

```
