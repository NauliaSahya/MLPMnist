import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import random

# Load and split images
try:
    A = Image.open("MnistExamples.png").convert('L')
except FileNotFoundError:
    print("Error: File 'MnistExamples.png' not found.")
    exit()

AGray = np.array(A)

# Initial coordinates and spacing
firstRow = 12
firstCol = 28
dataRowNum = 28
dataColNum = 27
xSpace = np.array([8, 8, 9, 8, 8, 8, 9, 8, 8, 8, 9, 8, 8, 8, 9, 8])
ySpace = np.array([5, 5, 5, 5, 6, 5, 5, 5, 5, 5])
totalImages = []

# Dynamic position variables
currentRow = firstRow
for kk in range(10):
    currentCol = firstCol
    for ii in range(16):
        # Extract image using current position
        img = AGray[currentRow : currentRow + dataRowNum, currentCol : currentCol + dataColNum]
        totalImages.append(img)
        
        # Update column position for next image
        currentCol += dataColNum + xSpace[ii]
        
    # Update row position for next row of images
    currentRow += dataRowNum + ySpace[kk]

totalImages = np.array(totalImages)

# Prepare data
numImages = totalImages.shape[0]
inputSize = 28 * 27
outputSize = 10
actualLabels = np.repeat(np.arange(10), 16)

# Flatten and normalize images
X = np.zeros((inputSize, numImages))
for i in range(numImages):
    img = totalImages[i, :, :]
    X[:, i] = img.flatten() / 255.0

# Create one-hot encoded labels
Y = np.zeros((outputSize, numImages))
for i in range(numImages):
    Y[actualLabels[i], i] = 1

# Split data into training and testing sets
test_per_digit = 3
test_indices = []

for digit in range(10):
    digit_indices = np.where(actualLabels == digit)[0]
    selected_indices = np.random.choice(digit_indices, test_per_digit, replace=False)
    test_indices.extend(selected_indices)

test_indices = sorted(test_indices)
all_indices = np.arange(numImages)
train_indices = np.setdiff1d(all_indices, test_indices)

X_train = X[:, train_indices]
X_test = X[:, test_indices]
train_labels = actualLabels[train_indices]
test_labels = actualLabels[test_indices]
Y_train = Y[:, train_indices]
Y_test = Y[:, test_indices]

print(f'Jumlah data training: {len(train_indices)}')
print(f'Jumlah data testing: {len(test_indices)}')

# MLP Parameters
hidden_layers = [25]
alpha = 1
learning_rate = 0.01
epochs = 1000

layer_sizes = [inputSize] + hidden_layers + [outputSize]
num_layers = len(layer_sizes)

# Initialize weights and biases
W = []
b = []
for i in range(num_layers - 1):
    W.append(np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i]))
    b.append(np.zeros((layer_sizes[i+1], 1)))

# Training loop
start_train_time = time.time()
accuracy_history = []

for epoch in range(epochs):
    correct_predictions = 0
    for img_idx in range(len(train_indices)):
        # Forward propagation
        a = [None] * num_layers
        z = [None] * (num_layers - 1)
        a[0] = X_train[:, img_idx].reshape(-1, 1)

        for layer in range(num_layers - 1):
            z[layer] = W[layer] @ a[layer] + b[layer]
            a[layer+1] = 1.0 / (1.0 + np.exp(-alpha * z[layer]))
        
        predicted_idx = np.argmax(a[-1])
        actual_idx = train_labels[img_idx]
        if predicted_idx == actual_idx:
            correct_predictions += 1
        
        # Backpropagation
        delta = [None] * (num_layers - 1)
        
        error_output = a[-1] - Y_train[:, img_idx].reshape(-1, 1)
        delta[-1] = error_output * (alpha * a[-1] * (1 - a[-1]))
        
        for layer in range(num_layers - 3, -1, -1):
            delta[layer] = (W[layer+1].T @ delta[layer+1]) * (alpha * a[layer+1] * (1 - a[layer+1]))
            
        # Update weights and biases
        for layer in range(num_layers - 1):
            W[layer] -= learning_rate * (delta[layer] @ a[layer].T)
            b[layer] -= learning_rate * delta[layer]
            
    accuracy_history.append((correct_predictions / len(train_indices)) * 100)
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{epochs} - Training Accuracy: {accuracy_history[-1]:.2f}%')

end_train_time = time.time()
time_train = end_train_time - start_train_time

# Test the model
def testModel(W, b, X, actualLabels, alpha):
    numImages = X.shape[1]
    predictions = []
    correct_predictions = 0

    start_time = time.time()
    for img_idx in range(numImages):
        a = [None] * len(W)
        z = [None] * len(W)
        a[0] = X[:, img_idx].reshape(-1, 1)

        for layer in range(len(W)):
            z[layer] = W[layer] @ a[layer] + b[layer]
            if layer < len(W) - 1:
                a[layer+1] = 1.0 / (1.0 + np.exp(-alpha * z[layer]))
            else: # Output layer
                a.append(1.0 / (1.0 + np.exp(-alpha * z[layer])))
        
        predicted_label = np.argmax(a[-1])
        predictions.append(predicted_label)
        
        if predicted_label == actualLabels[img_idx]:
            correct_predictions += 1
            
    total_time = time.time() - start_time
    accuracy = (correct_predictions / numImages) * 100
    return np.array(predictions), accuracy, total_time

test_predictions, test_accuracy, total_time = testModel(W, b, X_test, test_labels, alpha)

print(f'\nFinal Training Accuracy: {accuracy_history[-1]:.2f}%')
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Total waktu training: {time_train:.4f} detik')
print(f'Total waktu testing: {total_time:.4f} detik')
print(f'Waktu testing tiap gambar: {total_time / len(test_labels):.4f} detik')

# Plot training accuracy
plt.figure()
plt.plot(accuracy_history)
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.show()

# Display test results
plt.figure(figsize=(15, 5))
plt.suptitle('Test Results (Actual vs Predicted)', fontsize=16)
for i in range(len(test_labels)):
    plt.subplot(3, 10, i + 1)
    img_index = test_indices[i]
    plt.imshow(totalImages[img_index], cmap='gray')
    plt.title(f'A:{test_labels[i]} P:{test_predictions[i]}', fontsize=8)
    plt.axis('off')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Calculate and display confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(test_labels, test_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()