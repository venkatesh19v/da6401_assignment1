import numpy as np
import argparse
from optimizer import SGD, Momentum, NAGD, RMSProp, Adam, Nadam
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def parse_args():
    parser = argparse.ArgumentParser()
        
    # Dataset arguments
    parser.add_argument('--dataset', type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help='Dataset to use')
    
    # Training configuration arguments
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--loss', type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help='Loss function')
    parser.add_argument('--optimizer', type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd", help='Optimizer')
    
    # Learning and optimization parameters
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='Momentum (for momentum and nag optimizers)')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta (for RMSprop)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 (for Adam and Nadam)')
    parser.add_argument('--beta2', type=float, default=0.5, help='Beta2 (for Adam and Nadam)')
    parser.add_argument('--epsilon', type=float, default=0.000001, help='Epsilon for numerical stability')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimization')
    parser.add_argument('--weight_init', type=str, choices=["random", "xavier"], default="random", help='Weight initialization method')
    
    # Neural network structure parameters
    parser.add_argument('--num_layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default=4, help='Number of neurons in each hidden layer')
    parser.add_argument('--activation', type=str, choices=["identity", "sigmoid", "tanh", "relu"], default="sigmoid", help='Activation function')
    return parser.parse_args()  # Fixed typo here


def get_optimizer(args):
    if args.optimizer == "sgd":
        return SGD(learning_rate=args.learning_rate)
    elif args.optimizer == "momentum":
        return Momentum(learning_rate=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == "nag":
        return NAGD(learning_rate=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == "rmsprop":
        return RMSProp(learning_rate=args.learning_rate, beta=args.beta)
    elif args.optimizer == "adam":
        return Adam(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)
    elif args.optimizer == "nadam":
        return Nadam(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)


def plot_confusion_matrix(y_true, y_pred, labels, loss):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f"Confusion Matrix: {loss}")
    args = parse_args()
    plt.savefig(f"plot_confusion_matrix_{args.loss}_{args.optimizer}_{args.activation}_lr{args.learning_rate}_batch{args.batch_size}.png")
    wandb.log({"Confusion Matrix": wandb.Image(plt)})
    plt.close()
