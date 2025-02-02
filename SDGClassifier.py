import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv("data.csv")

# Convert bool to binary
df['Residue'] = df['Residue'].astype(int) 

# Define features and target
x = df[['x', 'y', 'R', 'G', 'B']]
y = df['Residue'] 

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Initialize the SGDClassifier
sgd_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, max_iter=5000, learning_rate='optimal', early_stopping=True)

# Predict the labels for the test set
y_pred = sgd_clf.predict(x_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

import joblib

# Save the model and scaler
joblib.dump(sgd_clf, 'sgd_classifier_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load model and scaler
sgd_clf = joblib.load('sgd_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')

# Predict on new pixel
new_pixel = pd.DataFrame([[10, 20, 130, 145, 190]], columns=['x', 'y', 'R', 'G', 'B'])

# Scale and predict
new_pixel_scaled = scaler.transform(new_pixel)
prediction = sgd_clf.predict(new_pixel_scaled)

print(f"Predicted Class: {'Residue' if prediction[0] == 1 else 'Background'}")


