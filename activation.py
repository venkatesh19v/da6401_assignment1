import numpy as np

def relu(x):
    # ReLU: f(x) = max(0,x) (0 to +ve infinite) 
    relu = np.maximum(0, x)
    return relu

def relu_derivative(x):
    # ReLU Derivative: f(x)= {0 if x<0; 1 if x>0}  | (0. to 1.)
    relu_derivative = (x > 0).astype(float) 
    return relu_derivative

def sigmoid(x):
    # Sigmoid: f(x) = 1 / (1 + e^(-x)) | (0 to 1)
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid

def sigmoid_derivative(x):
    # Sigmoid Derivative: f(x) = 1 / (1 + e^(-x)) * (1 - f(x)) | (0 to +ve infinite)
    sig = sigmoid(x)
    sigmoid_derivative = sig * (1 - sig)
    return sigmoid_derivative

def tanh(x):
    # Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x)) | (-1 to 1)
    tanh = np.tanh(x)
    return tanh

def tanh_derivative(x):
    # Tanh Derivative: f(x) = 1 - f(x)^2 | (0 to 1) 
    tanh_derivative = 1 - np.tanh(x)**2
    return tanh_derivative

def softmax(x):
    # Softmax: f(x) = e^(x - max(x)) / sum(e^(x - max(x))) | (0 to 1)
    ex_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    soft_max = ex_x / np.sum(ex_x, axis=1, keepdims=True)
    return soft_max