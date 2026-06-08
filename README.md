# Kidney Stone Detection Using Convolutional Neural Networks (CNN)

This project implements a Deep Learning model utilizing a Convolutional Neural Network (CNN) to automatically detect kidney stones from CT scan images. The model classifies scans into two categories: **Kidney Stone** and **Normal**.

---

## 📌 Project Overview
Kidney stones (nephrolithiasis) are a common and painful urinary tract disorder. Early detection using medical imaging is crucial for timely diagnosis and treatment. This project employs computer vision techniques to preprocess CT scan images and trains a binary classification CNN model to achieve high diagnostic accuracy.

---

## 📊 Dataset Detail
* **Source:** Sourced from Kaggle dataset: [raagbhutani/kidneystone](https://www.kaggle.com/datasets/raagbhutani/kidneystone).
* **Classes:** 
  - `Kidney_stone`
  - `Normal`
* **Size:** 1,609 total images.
* **Pre-processing:**
  - Loaded as grayscale images.
  - Resized to `128 x 128` pixels.
  - Normalized pixel values to the range `[0, 1]`.
  - Augmented via `ImageDataGenerator` (rotation range = 20, zoom range = 0.2, width/height shifts = 0.1, horizontal flip = True) to reduce overfitting.
* **Train/Test Split:** 80% Training (1,287 images) and 20% Testing (322 images).

---

## 🧠 Model Architecture

The model is built using TensorFlow/Keras Sequential API:

| Layer Type | Configuration / Output Shape | Activation Function |
| :--- | :--- | :--- |
| **Conv2D** | 32 filters, (3,3) kernel, Input: (128, 128, 1) | ReLU |
| **MaxPooling2D** | (2,2) pool size | - |
| **Conv2D** | 64 filters, (3,3) kernel | ReLU |
| **MaxPooling2D** | (2,2) pool size | - |
| **Flatten** | Reshapes features to 1D vector | - |
| **Dense** | 128 units | ReLU |
| **Dropout** | Rate: 0.5 (regularization) | - |
| **Dense (Output)** | 1 unit (Binary output) | Sigmoid |

* **Optimizer:** Adam
* **Loss Function:** Binary Cross-Entropy
* **Metrics:** Accuracy

---

## 📈 Training Results & Performance

The model was trained for **10 epochs**. The final performance obtained is:
* **Training Accuracy:** ~97.90%
* **Validation Accuracy:** ~89.44%
* **Training Loss:** ~0.0645
* **Validation Loss:** ~0.2749

Both accuracy and loss trends show stable convergence during training.

---

## ⚙️ Prerequisites & Installation

To run this project, make sure you have Python installed, then install the necessary dependencies:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn kagglehub
```

---

## 🚀 How to Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/DevSheta07/Kidney-Stone-Detection-Using-CNN.git
   cd Kidney-Stone-Detection-Using-CNN
   ```

2. **Run the Jupyter Notebook:**
   Open and execute the cells in `KS.ipynb` to download the dataset, preprocess images, train the CNN model, and evaluate performance.
   ```bash
   jupyter notebook KS.ipynb
   ```

3. **Using Saved Model:**
   The trained model weights are saved in `kidney_stone_model.h5`. You can load the model directly in Python for inference:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('kidney_stone_model.h5')
   ```

---

## 📂 Project Structure

```
├── CT_SCAN/                  # Dataset directory containing images sorted into class folders
├── KS.ipynb                  # Main Jupyter Notebook detailing training and validation
├── kidney_stone_model.h5      # Pre-trained CNN model weights
├── README.md                 # Project documentation
└── 23BCP110_AI_Project_Report.docx  # Detailed project report
```
