import numpy as np
from PIL import Image
import time
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function to test the model
def testModel(W, b, X, actualLabels, alpha):
    num_layers = len(W) + 1
    numImages = X.shape[1]
    predictions = []
    correct_predictions = 0
    
    for img_idx in range(numImages):
        a = [None] * num_layers
        z = [None] * (num_layers - 1)
        a[0] = X[:, img_idx].reshape(-1, 1)

        for layer in range(num_layers - 1):
            z[layer] = W[layer] @ a[layer] + b[layer]
            a[layer+1] = 1.0 / (1.0 + np.exp(-alpha * z[layer]))
        
        predicted_label = np.argmax(a[-1])
        predictions.append(predicted_label)
        
        if predicted_label == actualLabels[img_idx]:
            correct_predictions += 1
            
    accuracy = (correct_predictions / numImages) * 100
    return np.array(predictions), accuracy

# Load and split images
try:
    A = Image.open("MnistExamples.png").convert('L')
except FileNotFoundError:
    print("Error: File 'MnistExamples.png' not found.")
    exit()

AGray = np.array(A)

firstRow = 12
firstCol = 28
dataRowNum = 28
dataColNum = 27
xSpace = np.array([8, 8, 9, 8, 8, 8, 9, 8, 8, 8, 9, 8, 8, 8, 9, 8])
ySpace = np.array([5, 5, 5, 5, 6, 5, 5, 5, 5, 5])
totalImages = []

currentRow = firstRow
for kk in range(10):
    currentCol = firstCol
    for ii in range(16):
        img = AGray[currentRow : currentRow + dataRowNum, currentCol : currentCol + dataColNum]
        totalImages.append(img)
        currentCol += dataColNum + xSpace[ii]
    currentRow += dataRowNum + ySpace[kk]

totalImages = np.array(totalImages)
numImages = totalImages.shape[0]
inputSize = dataRowNum * dataColNum
outputSize = 10
actualLabels = np.repeat(np.arange(10), 16)

# Flatten and normalize
X = np.zeros((inputSize, numImages))
for i in range(numImages):
    X[:, i] = totalImages[i, :, :].flatten() / 255.0

# One-hot encoding labels
Y = np.zeros((outputSize, numImages))
for i in range(numImages):
    Y[actualLabels[i], i] = 1

# Split data
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

# MLP Parameters
hidden_layers = [128, 32]
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
        a = [None] * num_layers
        z = [None] * (num_layers - 1)
        a[0] = X_train[:, img_idx].reshape(-1, 1)

        for layer in range(num_layers - 1):
            z[layer] = W[layer] @ a[layer] + b[layer]
            a[layer+1] = 1.0 / (1.0 + np.exp(-alpha * z[layer]))

        # Count correct predictions for this epoch
        predicted_idx = np.argmax(a[-1])
        actual_idx = train_labels[img_idx]
        if predicted_idx == actual_idx:
            correct_predictions += 1
        
        delta = [None] * (num_layers - 1)
        error_output = a[-1] - Y_train[:, img_idx].reshape(-1, 1)
        delta[-1] = error_output * (alpha * a[-1] * (1 - a[-1]))
        
        for layer in range(num_layers - 3, -1, -1):
            delta[layer] = (W[layer+1].T @ delta[layer+1]) * (alpha * a[layer+1] * (1 - a[layer+1]))
            
        for layer in range(num_layers - 1):
            W[layer] -= learning_rate * (delta[layer] @ a[layer].T)
            b[layer] -= learning_rate * delta[layer]
            
    # Calculate and display training accuracy
    train_accuracy = (correct_predictions / len(train_indices)) * 100
    accuracy_history.append(train_accuracy)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs} - Training Accuracy: {train_accuracy:.2f}%")
            
end_train_time = time.time()
print(f"Training finished in {end_train_time - start_train_time:.2f} seconds.")

# Test the model on the test data
test_predictions, test_accuracy = testModel(W, b, X_test, test_labels, alpha)
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Plot training accuracy
plt.figure()
plt.plot(accuracy_history)
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.show()

# Display confusion matrix
cm = confusion_matrix(test_labels, test_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Save the trained model
model_data = {
    'W': W,
    'b': b,
    'alpha': alpha
}
with open('mlp_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("Model saved to 'mlp_model.pkl'.")