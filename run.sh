#!/bin/bash
echo "Running Test 1: Minimal training run with default parameters..."
python3 train.py --epochs 15 --weight_init xavier --weight_decay 0 --batch_size 8 --optimizer nadam --learning_rate 0.0001 --activation relu --num_layers 4 --hidden_size 256

echo "Running Test 2: Minimal training run with default parameters..."
python3 train.py --epochs 15 --weight_init xavier --weight_decay 0.0005 --batch_size 32 --optimizer adam --learning_rate 0.0001 --activation tanh --num_layers 4 --hidden_size 256
# relu	32	0.9	20	0.000001	3	256	784	0.0001	0.95		nadam	10	0.0005	xavier
