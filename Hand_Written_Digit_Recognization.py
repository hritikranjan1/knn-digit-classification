import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
digits = load_digits()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Create the K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Let's visualize some predictions
n_samples = 10
sample_images = X_test[:n_samples]
sample_labels = y_pred[:n_samples]

plt.figure(figsize=(8, 3))
for i in range(n_samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images[i].reshape((8, 8)), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Predicted: %d" % sample_labels[i])

plt.show()