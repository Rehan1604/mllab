import numpy as np

# Input image (5x5)
image = np.array([
    [1, 1, 0, 0, 1],
    [1, 1, 0, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0],
    [1, 1, 0, 0, 1]
], dtype=float)

# 3x3 convolution filter
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=float)

# Convolution
def convolution(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    output = np.zeros((h-kh+1, w-kw+1))

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = img[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output

# ReLU
def relu(x):
    return np.maximum(0, x)

# Max pooling
def max_pooling(x):
    h, w = x.shape
    output = np.zeros((h//2, w//2))

    for i in range(0, h-1, 2):
        for j in range(0, w-1, 2):
            output[i//2, j//2] = np.max(x[i:i+2, j:j+2])

    return output

# Fully connected layer
feature_map = convolution(image, kernel)
activated = relu(feature_map)
pooled = max_pooling(activated)
flattened = pooled.flatten()

weights = np.random.randn(len(flattened), 2)
bias = np.zeros(2)
output = np.dot(flattened, weights) + bias

# Softmax
exp_values = np.exp(output - np.max(output))
probabilities = exp_values / np.sum(exp_values)

print("Convolution:\n", feature_map)
print("After ReLU:\n", activated)
print("After Max Pooling:\n", pooled)
print("Flattened:", flattened)
print("Class probabilities:", probabilities)
print("Predicted class:", np.argmax(probabilities))
