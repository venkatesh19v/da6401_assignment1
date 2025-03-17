#!/bin/bash
echo "Running Test 1: Minimal training run with best parameters..."
python3 train.py --epochs 20 --weight_init random --weight_decay 0.5 --batch_size 64 --optimizer adam --learning_rate 0.001 --activation relu --num_layers 3 --hidden_size 256

echo "Running Test 2: Minimal training run with best parameters..."
python3 train.py --epochs 20 --weight_init xavier --weight_decay 0.0005 --batch_size 64 --optimizer rmsprop --learning_rate 0.001 --activation relu --num_layers 7 --hidden_size 256

echo "Running Test 3: Minimal training run with best parameters..."
python3 train.py --epochs 20 --weight_init xavier --weight_decay 0.0005 --batch_size 8 --optimizer nadam --learning_rate 0.0001 --activation relu --num_layers 7 --hidden_size 256

echo "Running Test 4: Minimal training run with best parameters for MSE..."
python3 train.py --epochs 10 --weight_init xavier --weight_decay 0.0005 --batch_size 8 --optimizer nadam --learning_rate 0.0001 --activation relu --num_layers 5 --hidden_size 256 --loss mean_squared_error

echo "Running Test 5: Minimal training run with best parameters for MSE..."
python3 train.py --epochs 15 --weight_init xavier --weight_decay 0.0005 --batch_size 8 --optimizer nadam --learning_rate 0.0001 --activation relu --num_layers 6 --hidden_size 256 --loss mean_squared_error

echo "Running Test 6: Minimal training run with best parameters for MSE..."
python3 train.py --epochs 20 --weight_init xavier --weight_decay 0.0005 --batch_size 8 --optimizer nadam --learning_rate 0.0001 --activation tanh --num_layers 7 --hidden_size 256 --loss mean_squared_error

echo "Running Test 7: Minimal training run with best parameters on mnist dataset ..."
python3 train.py --epochs 20 --weight_init random --weight_decay 0.5 --batch_size 64 --optimizer adam --learning_rate 0.001 --activation relu --num_layers 3 --hidden_size 256 --dataset mnist

echo "Running Test 8: Minimal training run with best parameters on mnist dataset for MSE..."
python3 train.py --epochs 20 --weight_init random --weight_decay 0.5 --batch_size 64 --optimizer adam --learning_rate 0.001 --activation relu --num_layers 3 --hidden_size 256 --dataset mnist --loss mean_squared_error