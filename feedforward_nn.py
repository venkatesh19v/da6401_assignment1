import numpy as np
from activation import relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_derivative, softmax


class FeedForwardNN:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', weight_init='random'):
        self.activation_name = activation
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        else:
            raise ValueError("Unsupported activation function.")
        
        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            if weight_init == "random":
                W = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.01
            elif weight_init == "xavier":
                W = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(1.0 / self.layer_sizes[i])
            else:
                print(f"Invalid weight_init: {weight_init}")
                raise ValueError("Unsupported weight initialization method.")

            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)
    
    def forward(self, X):
        cache = {"A": [], "Z": []}
        A = X
        cache["A"].append(A)
        
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            cache["Z"].append(Z)
            A = self.activation(Z)
            cache["A"].append(A)
        
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        cache["Z"].append(Z)
        A = softmax(Z)
        cache["A"].append(A)
        return A, cache

    def backward(self, X, Y_true, cache):
        m = X.shape[0]
        L = len(self.weights)
        grads_W = [None] * L
        grads_b = [None] * L

        A_final = cache["A"][-1]
        dZ = A_final - Y_true
        grads_W[L - 1] = np.dot(cache["A"][-2].T, dZ) / m
        grads_b[L - 1] = np.sum(dZ, axis=0, keepdims=True) / m

        for i in range(L-2, -1, -1):
            dA = np.dot(dZ, self.weights[i + 1].T)
            Z = cache["Z"][i]
            dZ = dA * self.activation_derivative(Z)
            grads_W[i] = np.dot(cache["A"][i].T, dZ) / m
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True) / m

        return grads_W, grads_b

    def backward_mse(self, X, Y_true, cache):
        m = X.shape[0]
        L = len(self.weights)
        grads_W = [None] * L
        grads_b = [None] * L

        # MSE loss: loss = (1/(2*m)) * sum((A_final - Y_true)^2)
        # dL/dA_final = (A_final - Y_true) / m
        A_final = cache["A"][-1]
        dZ = (A_final - Y_true) / m
        grads_W[L - 1] = np.dot(cache["A"][-2].T, dZ)
        grads_b[L - 1] = np.sum(dZ, axis=0, keepdims=True)

        # Propagate gradients backwards through the hidden layers
        for i in range(L - 2, -1, -1):
            dA = np.dot(dZ, self.weights[i + 1].T)
            Z = cache["Z"][i]
            dZ = dA * self.activation_derivative(Z)
            grads_W[i] = np.dot(cache["A"][i].T, dZ) / m
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True) / m

        return grads_W, grads_b


    def update_parameters(self, grads_W, grads_b, optimizer):
        params = self.weights + self.biases
        grads = grads_W + grads_b
        updated_params = optimizer.update(params, grads)
        L = len(self.weights)
        self.weights = updated_params[:L]
        self.biases = updated_params[L:]
    
    def cross_entropy_loss(self, Y_pred, Y_true):
        """
        Compute cross-entropy loss
        loss =  - ∑ Y_true.log(Y_pred)
        """
        m = Y_true.shape[0]
        loss = -np.sum(Y_true * np.log(Y_pred + 1e-8)) / m
        return loss
    
    def squared_error_loss(self, Y_pred, Y_true):
        """
        Computes the Mean Squared Error (MSE) loss.
        loss = 1/m * ∑ (Y_pred - Y_true)^2"""
        return np.mean(np.sum((Y_pred - Y_true) ** 2, axis=1))

    def compute_accuracy(self, Y_pred, Y_true):
        predictions = np.argmax(Y_pred, axis=1)
        labels = np.argmax(Y_true, axis=1)
        return np.mean(predictions == labels)


# python train.py --epochs 15 --weight_init xavier --weight_decay 0.0005 --batch_size 32 --optimizer adam --learning_rate 0.0001 --activation tanh --num_layers 4 --hidden_size 256
