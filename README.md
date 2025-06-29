# 🖼️ Simple Image Classifier (CIFAR-10)

This project implements a **Convolutional Neural Network (CNN)** to classify images in the **CIFAR-10 dataset**, a standard computer vision dataset containing 10 classes of 32x32 color images.



## 📑 Table of Contents

* [Introduction](#introduction)
* [Dataset](#dataset)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Project Structure](#project-structure)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)



## 📝 Introduction

**CIFAR-10** is a popular dataset for benchmarking image classification models. It consists of **60,000 32x32 color images** in 10 classes, with 6,000 images per class. This project trains a **simple CNN model** to classify these images into their respective categories.



## 📚 Dataset

* **Dataset:** [CIFAR-10]()
* **Classes (10 total):**

  * Airplane
  * Automobile
  * Bird
  * Cat
  * Deer
  * Dog
  * Frog
  * Horse
  * Ship
  * Truck



## ✨ Features

✅ Load and preprocess CIFAR-10 dataset

✅ Build CNN model using Keras (TensorFlow backend)

✅ Train the model with validation

✅ Evaluate model performance on test data

✅ Visualize sample predictions with actual and predicted labels



## 🛠️ Technologies Used

* **Python 3**
* **TensorFlow / Keras**
* `numpy`
* `matplotlib`
* `seaborn`
* **Jupyter Notebook**



## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Simple-Image-Classifier-CIFAR-10.git
cd Simple-Image-Classifier-CIFAR-10
```

2. **Create and activate a virtual environment (optional)**

```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**

```bash
jupyter notebook
```



## ▶️ Usage

1. Open `CIFAR10_Image_Classifier.ipynb` in Jupyter Notebook.
2. Run cells sequentially to:

   * Import libraries and load dataset
   * Normalize image pixel values
   * Build and compile the CNN model
   * Train the model on training data
   * Evaluate the model on test data
   * Visualize predictions on sample test images



## 🏗️ Model Architecture

Sample CNN architecture used:

* **Conv2D Layer:** 32 filters, (3x3) kernel, ReLU activation
* **MaxPooling2D Layer:** (2x2) pool size
* **Conv2D Layer:** 64 filters, (3x3) kernel, ReLU activation
* **MaxPooling2D Layer:** (2x2) pool size
* **Flatten Layer**
* **Dense Layer:** 64 units, ReLU activation
* **Output Dense Layer:** 10 units (classes), Softmax activation

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```



## 📁 Project Structure

```
Simple-Image-Classifier-CIFAR-10/
 ┣ CIFAR10_Image_Classifier.ipynb
 ┣ requirements.txt
 ┗ README.md
```



## 📈 Results

* **Training Accuracy:** *e.g. 80%*
* **Test Accuracy:** *e.g. 75%*

The model demonstrates strong performance in classifying CIFAR-10 images, showcasing the power of CNNs for basic computer vision tasks.



## 📊 Example Prediction

```python
import numpy as np

# Predict on a single image
index = 25
img = x_test[index].reshape(1,32,32,3)
prediction = model.predict(img)
predicted_label = np.argmax(prediction)
print("Predicted class:", class_names[predicted_label])
```



## 🤝 Contributing

Contributions are welcome to:

* Enhance model accuracy with deeper architectures
* Implement data augmentation for better generalization
* Experiment with transfer learning using pretrained models (e.g., VGG, ResNet)
* Deploy as a web app for live image classification

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request



## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.



## 📬 Contact

**Ugama Benedicta Kelechi**
[LinkedIn](www.linkedin.com/in/ugama-benedicta-kelechi-codergirl-103041300) | [Email](mailto:ugamakelechi501@gmail.com) | [Portfolio](#)



### ⭐️ If you find this project useful, please give it a star!

