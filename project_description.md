## Project 11: *Emotion Recognition from Facial Landmarks*

### 🧠 Motivation

The [FER‑2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013/data) contains ~30,000 grayscale face images (48×48) labeled with seven basic emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). Learning emotion from such a small dataset is challenging—direct pixel models work, but structured geometric features (like **facial landmarks**) often improve performance and interpretability.

This project invites you to explore both **handcrafted geometric features** and **neural network models**, comparing classical and deep approaches.

### 🎯 Objectives

- Extract **facial landmarks** from images (e.g., 68-point features) using [Dlib](https://github.com/davisking/dlib)
- Design and extract geometric features of faces (e.g., distances between facial landmarks, angles, ratios, etc.)
- Alternatively, train **MLP**s or simple **CNN**s directly on the raw images
- Compare performance, **interpretability**, and computational cost

### 📊 Dataset

[**FER‑2013** available on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013/data):

- 48×48 grayscale face crops, split into Training (~29 k), Public Test (~3.6 k)
- Seven emotion with labels 0–6

### 🛠️ Suggested Models/Tools

#### 1. Landmark-based feature extraction
The dlib library contains a pretrained facial landmark detector (`shape_predictor_68_face_landmarks.dat`) used to estimate the location of 68 coordinate pairs of facial structures. You can find an easy code example for applying this detector in [this GitHub repo](https://github.com/Practical-CV/Facial-Landmarks-Detection-with-DLIB/tree/master), as well as in this [tutorial on extracting facial landmarks](https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/).

You can extract facial feature as folliws: 
- Extract 68 facial landmarks using Dlib’s detector
- Design and compute features such as:
  - Pairwise Euclidean distances between landmarks
  - Ratios of eye/mouth width, eyebrow height

> **Note:** The dataset is provided as raw images organized in folders by class. You can use `cv2` (as in the GitHub example above) or `scikit-image` packages to load these images. In particular, `skimage.io.ImageCollection` provides tools to load images from a base folder. 

#### 2. NN baseline 
 
You may use MLPs (implemented in the course) or implement a simple Convolutional NN (CNN) and apply it to raw image pixel values as a baseline. 

### ✅ Minimum Requirements

- Preprocess FER‑2013 images, extract facial landmarks, and compute several geometric features
- Train and evaluate at least one classical model (LR/SVM) on landmarks
- Compare to NN models on raw pixels
- Report and visualize results, e.g.:
  - Test accuracy for each approach
  - Confusion matrices
  - Visual summary of key geometric features that differentiate emotions

### 🚀 Optional Extensions

- Try and compare more geometric features based on landmarks
- Implement a CNN for comparison

### 📚 References

- Casado & Lopez (2021) – [Real‐time face alignment: evaluation methods, training strategies and implementation optimization](https://link.springer.com/content/pdf/10.1007/s11554-021-01107-w.pdf)  
- Sagonas et al. (2016) – [300 faces In-the-wild challenge: Database and results](https://core.ac.uk/download/pdf/189325203.pdf)  
- [Tutorial "Facial landmarks with dlib, OpenCV, and Python"](https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)


