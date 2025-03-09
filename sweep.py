
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import wandb
import time
from optimizer import SGD, Momentum, NAGD, RMSProp, Adam, Nadam
from feedforward_nn import FeedForwardNN
from utils import one_hot_encoding

projectId = "DA6401_Assignment1"

# Initializing wandb
def init_wandb():
    try:
        wandb.init(project=projectId,
            notes="This is Assignment 1 for the course DA6401", 
            tags=["assignment_1", "fashion_mnist"])
    except BrokenPipeError:
        time.sleep(5)  
        wandb.init()

# Function to get the optimizer
def get_optimizer(name):
    if name == "sgd":
        return SGD(learning_rate=wandb.config.learning_rate)
    elif name == "momentum":
        momentum = wandb.config.get('momentum', 0.9)  
        return Momentum(learning_rate=wandb.config.learning_rate, momentum=momentum)
    elif name == "nagd":
        momentum = wandb.config.get('momentum', 0.9)  
        return NAGD(learning_rate=wandb.config.learning_rate, momentum=momentum)
    elif name == "rmsprop":
        beta = wandb.config.get('beta', 0.9)  
        epsilon = wandb.config.get('epsilon', 1e-8)  
        
        return RMSProp(learning_rate=wandb.config.learning_rate, beta=beta, epsilon=epsilon)
    elif name == "adam":
        beta1 = wandb.config.get('beta1', 0.9)  
        beta2 = wandb.config.get('beta2', 0.999)  
        epsilon = wandb.config.get('epsilon', 1e-8)  
        return Adam(learning_rate=wandb.config.learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    elif name == "nadam":
        beta1 = wandb.config.get('beta1', 0.9)  
        beta2 = wandb.config.get('beta2', 0.999)  
        epsilon = wandb.config.get('epsilon', 1e-8)  
        return Nadam(learning_rate=wandb.config.learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    else:
        raise ValueError("Unsupported optimizer.")

# Function to run the sweep in wandb
def train_sweep():
    init_wandb()  

    config = wandb.config
    run_name = f"hl_{config.hidden_layers}_hs_{config.hidden_size}_ep_{config.epochs}_bs_{config.batch_size}_lr_{config.learning_rate}_wi_{config.weight_init}_wd_{config.weight_decay}_op_{config.optimizer}_ac_{config.activation.lower()}"
    print(run_name)

    # Config to run sweep wandb 
    wandb.run.name = run_name
    input_size = config.input_size
    output_size = config.output_size
    hidden_layers = config.hidden_layers  
    hidden_size = config.hidden_size  
    activation_function = config.activation  
    weight_init = config.weight_init  
    optimizer_name = config.optimizer 
    batch_size = config.batch_size  
    num_epochs = config.epochs  
    loss_fn = config.loss_function
    print(f"loss_function_used: ",loss_fn)
    
    # Load Fashion-MNIST data (using the standard train/test split)
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Preprocess the images: flatten and normalize to [0, 1]
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    # Create a validation split: 90% training, 10% validation
    num_train = int(0.9 * X_train.shape[0])
    permutation = np.random.permutation(X_train.shape[0])
    X_train = X_train[permutation]
    y_train = y_train[permutation]
    X_train_split = X_train[:num_train]
    X_val_split = X_train[num_train:]
    y_train_split = y_train[:num_train]
    y_val_split = y_train[num_train:]

    num_classes = output_size  # classes 10
    Y_train_split = one_hot_encoding(y_train_split, num_classes)
    Y_val_split = one_hot_encoding(y_val_split, num_classes)
    hidden_layers = [hidden_size] * hidden_layers  # Creating hidden layers

    # Instantiating the model FeedForwardNN with backprop (Question 3)
    model = FeedForwardNN(input_size=input_size,output_size=output_size, hidden_layers=hidden_layers,
        activation=activation_function.lower(), weight_init=weight_init.lower())
    
    optimizer = get_optimizer(optimizer_name)
    num_samples = X_train_split.shape[0]
    num_batches = num_samples // batch_size

    best_val_loss = float('inf')
    best_epoch = -1

    # Training loop
    for epoch in range(num_epochs):
        # Shuffling training data at each epoch
        permutation = np.random.permutation(num_samples)
        X_train_epoch = X_train_split[permutation]
        Y_train_epoch = Y_train_split[permutation]
        epoch_loss = 0.0

        # Looping through the batches
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_train_epoch[start:end]
            Y_batch = Y_train_epoch[start:end]

            # Forward pass and compute loss
            Y_pred, cache = model.forward(X_batch)
            if loss_fn == "cross_entropy":
                loss = model.cross_entropy_loss(Y_pred, Y_batch)
                # Backward pass to compute gradients
                grads_W, grads_b = model.backward(X_batch, Y_batch, cache)

            else:
               loss = model.squared_error_loss(Y_pred, Y_batch)
               # Backward pass to compute gradients for mse
               grads_W, grads_b = model.backward_mse(X_batch, Y_batch, cache)


            epoch_loss += loss

            # Update parameters using the chosen optimizer
            model.update_parameters(grads_W, grads_b, optimizer)

        avg_loss = epoch_loss / num_batches   
        # training_loss = avg_loss

        # Evaluating on the validation set
        Y_val_pred, _ = model.forward(X_val_split)
        
        if loss_fn == "cross_entropy":
            val_loss = model.cross_entropy_loss(Y_val_pred, Y_val_split)
        else:
            val_loss = model.squared_error_loss(Y_val_pred, Y_val_split)

        # val_loss = model.cross_entropy_loss(Y_val_pred, Y_val_split)

        predictions = np.argmax(Y_val_pred, axis=1)
        correct = np.sum(predictions == y_val_split)
        val_accuracy_epoch = correct / len(y_val_split)

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_epoch = epoch + 1


        # Log epoch-level metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy_epoch,
        })

    print(f"Sweep run complete: Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}, Val Accuracy: {val_accuracy_epoch:.4f}")

sweep_config = {
    'method': 'bayes',  # bayes or random
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'name': 'hyperparameter_sweeping_mean_squared_error',
    'parameters': {
        'loss_function': {'value': 'mean_squared_error'},
        # 'epochs': {'values': [5, 10, 15, 20]},
        'epochs': {'values': [5, 10, 15]},
        'hidden_layers': {'values': [3, 4, 5, 6]},
        # 'hidden_layers': {'values': [3, 4, 5, 6, 7]},  
        'hidden_size': {'values': [32, 64, 128, 256]}, 
        'weight_decay': {'values': [0, 0.0005, 0.5]},  
        # 'learning_rate': {'values': [0.0001, 0.001, 0.01, 0.1]},
        'learning_rate': {'values': [0.0001, 0.001, 0.01]},
        'optimizer': {'values': ['sgd', 'momentum', 'nagd', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [8, 16, 32, 64]}, 
        'weight_init': {'values': ['random', 'xavier']},
        'activation': {'values': ['sigmoid', 'tanh', 'relu']}, 
        'input_size': {'value': 784},  
        'output_size': {'value': 10},  
        'epsilon':{ 'values': [1e-8, 1e-7, 1e-6]},
        'beta': {'values': [0.9, 0.99, 0.999]}, 
        'momentum': {'values': [0.8, 0.9, 0.95]},  
    }
}
sweep_id = wandb.sweep(sweep_config, project=projectId)

if __name__ == '__main__':
    wandb.agent(sweep_id, function=train_sweep, count=20)




