import numpy as np

class Optimizer:
    """
    Base Optimizer class that outlines the interface for optimization algorithms.
    params (list): List of model parameters (weights and biases).
    grads (list): List of gradients with respect to the model parameters.
    """
    def update(self, params, grads):
        raise NotImplementedError

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD)
    Formula:
        θ = θ - η * ∇θ
    where, 
        θ represents the model parameters (weights/biases),
        η is the learning rate, and 
        ∇θ is the gradient of the loss function.
    """
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
        return params

class Momentum(Optimizer):
    """
    Momentum optimizer class that helps accelerate SGD by considering past (history) gradients.
    Formula:
        v_t = β * v_(t-1) + ∇wt
    where,
        v_t is the velocity, 
        β is the momentum term, and 
        η is the learning rate.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]
        return params

class NAGD(Optimizer):
    """
    Nesterov Accelerated Gradient (NAG), improved version of momentum.
    Formula:
        v_t = β * v_(t-1) + ∇(wt - β * v_(t-1))   
    where, 
        v_t is the momentum term, 
        β is the momentum coefficient, 
        η is the learning rate.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for i in range(len(params)):
            v_prev = self.v[i].copy()
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += -self.momentum * v_prev + (1 + self.momentum) * self.v[i]
        return params

class RMSProp(Optimizer):
    """
    RMSProp - Root Mean Square Propagation, which adjusts the learning rate based on recent gradients' magnitude.
    Formula:
        S_t = β * S_(t-1) + (1 - β) * (∇wt)^2
        wt+1 = wt - η * ∇wt / sqrt(S_t + ε)
    where, 
        S_t is the moving average of squared gradients, 
        β is the decay term,
        η is the learning rate, and     
        ε is a small number to prevent division by zero.
    """
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.S = None

    def update(self, params, grads):
        if self.S is None:
            self.S = [np.zeros_like(p) for p in params]
        for i in range(len(params)):
            self.S[i] = self.beta * self.S[i] + (1 - self.beta) * (grads[i] ** 2)
            params[i] -= self.lr * grads[i] / (np.sqrt(self.S[i]) + self.epsilon)
        return params

class Adam(Optimizer):
    """
    Adam optimizer, which combines momentum and RMSProp to adapt learning rates for each parameter.

    Formula:
        m_t = β1 * m_(t-1) + (1 - β1) * ∇wt
        v_t = β2 * v_(t-1) + (1 - β2) * (∇wt))^2
        m_t_hat = m_t / (1 - β1^t)
        v_t_hat = v_t / (1 - β2^t)
        wt = wt - η * m_t_hat / (sqrt(v_t_hat) + ε)
    where,
        m_t and v_t are moment estimates, 
        β1 and β2 are exponential decay rates, and 
        t is the time step.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params

class Nadam(Optimizer):
    """
    Nadam (Nesterov-accelerated Adaptive Moment Estimation),

    Formula:
        m_t+1 = β1 * m_t + (1 - β1) * ∇wt
        v_t+1 = β2 * v_t + (1 - β2) * (∇wt)^2
        m_t_hat = m_t+1 / (1 - β1^t+1)
        v_t_hat = v_t+1 / (1 - β2^t+1)
        wt+1 = wt - (η / (sqrt(v_t_hat) + ε)* (β1 * m_t+1_hat + (1 - β1) * ∇wt/ 1 - β1_t+1)
    where, 
        β1 and β2 are exponential decay rates, and
        t is the time step.
    """    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            nesterov_m = (self.beta1 * m_hat) + ((1 - self.beta1) * grads[i] / (1 - self.beta1 ** self.t))
            params[i] -= self.lr * nesterov_m / (np.sqrt(v_hat) + self.epsilon)
        return params