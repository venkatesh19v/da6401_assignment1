import numpy as np
import wandb
import pickle
from feedforward_nn import FeedForwardNN
from utils import one_hot_encoding, parse_args, get_optimizer, plot_confusion_matrix
from keras.datasets import fashion_mnist, mnist

projectId = "DA6401_Assignment1"

# Load real Fashion-MNIST data
# if args.dataset == "fashion_mnist":
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
# elif args.dataset == "mnist":
# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
X_train, X_test = X_train.reshape(-1, 784) / 255.0, X_test.reshape(-1, 784) / 255.0
Y_train = one_hot_encoding(Y_train, 10)
Y_test = one_hot_encoding(Y_test, 10)

# Create validation set
val_split = 0.1
split_idx = int(X_train.shape[0] * (1 - val_split))
X_train, X_val = X_train[:split_idx], X_train[split_idx:]
Y_train, Y_val = Y_train[:split_idx], Y_train[split_idx:]

class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def train(model, optimizer, X_train, Y_train, X_val, Y_val, epochs, batch_size, loss):
    num_samples = X_train.shape[0]
    num_batches = num_samples // batch_size

    for epoch in range(epochs):
        permutation = np.random.permutation(num_samples)
        X_shuffled = X_train[permutation]
        Y_shuffled  = Y_train[permutation]
        
        epoch_loss, epoch_accuracy = 0.0, 0.0
        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            X_batch = X_shuffled[start:end]
            Y_batch = Y_shuffled[start:end]
            
            Y_pred, cache = model.forward(X_batch)

            if args.loss == "cross_entropy":
               loss = model.cross_entropy_loss(Y_pred, Y_batch)
               accuracy = model.compute_accuracy(Y_pred, Y_batch)
               grads_W, grads_b = model.backward(X_batch, Y_batch, cache)


            else:
               loss = model.squared_error_loss(Y_pred, Y_batch)
               accuracy = model.compute_accuracy(Y_pred, Y_batch)
               grads_W, grads_b = model.backward_mse(X_batch, Y_batch, cache)
                    
            model.update_parameters(grads_W, grads_b, optimizer)
            
            epoch_loss += loss
            epoch_accuracy += accuracy
        
        # Calculate average loss and accuracy
        avg_loss = epoch_loss / num_batches 
        avg_accuracy = epoch_accuracy / num_batches
        
        # Validation accuracy
        Y_val_pred, _ = model.forward(X_val)
        if args.loss == "cross_entropy":
            val_loss = model.cross_entropy_loss(Y_val_pred, Y_val)
        else:
            val_loss = model.squared_error_loss(Y_val_pred, Y_val)

        val_accuracy = model.compute_accuracy(Y_val_pred, Y_val)

        print("loss function",args.loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {avg_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": avg_loss, "accuracy": avg_accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy})

def test(model, X_test, Y_test):
    Y_pred, _ = model.forward(X_test)
    if args.loss == "cross_entropy":
        loss, accuracy = model.cross_entropy_loss(Y_pred, Y_test), model.compute_accuracy(Y_pred, Y_test)
    else:
        loss, accuracy = model.squared_error_loss(Y_pred, Y_test), model.compute_accuracy(Y_pred, Y_test)

    Y_pred_classes, Y_test_true = np.argmax(Y_pred, axis=1), np.argmax(Y_test, axis=1)
    
    wandb.log({"loss function": args.loss,"test_loss": loss, "test_accuracy": accuracy})
    print(f"Testing Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return Y_pred_classes, Y_test_true

def main(args):
    wandb.init(project=projectId, notes="Improved training script with real data", tags=["mnist", "feedforward_nn"])
    
    model = FeedForwardNN(input_size=784, output_size=10, hidden_layers=[args.hidden_size] * args.num_layers, activation=args.activation, weight_init=args.weight_init)
    optimizer = get_optimizer(args)

    train(model, optimizer, X_train, Y_train, X_val, Y_val, epochs=args.epochs, batch_size=args.batch_size, loss=args.loss)
    
    Y_pred_classes, Y_test_true = test(model, X_test, Y_test)
    plot_confusion_matrix(Y_test_true, Y_pred_classes, class_names, args.loss)
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=Y_test_true, preds=Y_pred_classes,
                        class_names=class_names)})

    
    model_name = f"fashion_mnist_{args.loss}_{args.optimizer}_{args.activation}_lr{args.learning_rate}_batch{args.batch_size}.pkl"
    with open(model_name, 'wb') as f:
        pickle.dump({'model': model, 'optimizer': optimizer}, f)
    print(f"Model saved as {model_name}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
