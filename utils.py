import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.metrics import confusion_matrix, jaccard_score

### Image loading ###

# Load image from given path
def load_image(image_path):
  return cv2.imread(image_path)

# Load and convert (255 -> 1) label image
def load_label(label_path):
  label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
  return (label == 255).astype(np.uint8)

### Feature extraction ###

# Extract normalized color histogram from an image
def extract_color_histogram(image):
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  hist_h = cv2.calcHist([hsv], [0], None, [256], [0, 256]).flatten()
  hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
  hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()
  return np.concatenate([hist_h, hist_s, hist_v]) / sum(hist_h + hist_s + hist_v)

# Extract histogram of gradient features from image
def extract_HOG_features(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
  return features


### Performance metrics

# Compute IoU score
def compute_iou(y_true, y_pred):
  return jaccard_score(y_true.flatten(), y_pred.flatten(), average='macro')

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
  cm = confusion_matrix(y_true, y_pred)
  plt.imshow(cm, cmap='Blues')
  plt.colorbar()
  plt.xlabel("Predicted")
  plt.ylabel("True")
  plt.xticks(np.arange(len(classes)), classes, rotation=45)
  plt.yticks(np.arange(len(classes)), classes)
  plt.show()





