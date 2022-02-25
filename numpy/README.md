# Training neural networks

## Install Libraries
```
pip install -r requirements.txt
```

## Usage

```bash
python .\train_fmnist.py -h
usage: train_fmnist.py [-h] [--model_dir MODEL_DIR] [--dataset DATASET]
                       [--model_name MODEL_NAME] [--batch_size BATCH_SIZE]
                       [--epochs EPOCHS] [--val_split VAL_SPLIT] [--seed SEED]
                       [--shuffle SHUFFLE] [--learning_rate LEARNING_RATE]
                       [--num_layers NUM_LAYERS] [--hidden_sizes HIDDEN_SIZES]
                       [--weight_decay_rate WEIGHT_DECAY_RATE]
                       [--optimizer OPTIMIZER] [--momentum MOMENTUM]
                       [--nesterov] [--weight_init WEIGHT_INIT]
                       [--activation ACTIVATION] [--loss LOSS] [--plot PLOT]
                       [--debug] [--wandb_project WANDB_PROJECT]
                       [--wandb_entity WANDB_ENTITY]

Train a model on the Fashion MNIST/ mnist dataset in cpu with numpy

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        Directory to store the models
  --dataset DATASET     fmnist
  --model_name MODEL_NAME
                        Name of the model
  --batch_size BATCH_SIZE
                        Batch size for training
  --epochs EPOCHS       Number of epochs to train for
  --val_split VAL_SPLIT
                        Validation split for training data
  --seed SEED           Random seed for training
  --shuffle SHUFFLE     Shuffle the training data
  --learning_rate LEARNING_RATE
                        Learning rate for training
  --num_layers NUM_LAYERS
                        Number of layers in the model
  --hidden_sizes HIDDEN_SIZES
                        this is an integer or a comma-separated list of
                        integers indicating the number of hidden units in each
                        layer of the model.
  --weight_decay_rate WEIGHT_DECAY_RATE
                        Weight decay rate (L2 regularizer) for training
  --optimizer OPTIMIZER
                        Optimizer to use for training
  --momentum MOMENTUM   Momentum for SGD/rmsprop optimizer
  --nesterov            Whether to use nesterov
  --weight_init WEIGHT_INIT
                        Weight initialization scheme
  --activation ACTIVATION
                        Activation function to use
  --loss LOSS           Loss function to use
  --plot PLOT           Plot the confusion matrix
  --debug               for debugging with limited data
  --wandb_project WANDB_PROJECT
                        W&B project name
  --wandb_entity WANDB_ENTITY
                        W&B entity name
```

Example commands to train your network:
```
python train_fmnist.py \
--activation=relu \
--batch_size=64 \
--dataset=fmnist \
--hidden_sizes=128 \
--learning_rate=0.001 \
--num_layers=4 \
--optimizer=nadam \
--wandb_entity=<wandb entity name> \
--wandb_project=dl-course \
--weight_decay_rate=0 \
--weight_init=xavier
```

Folder Structure:

```
├── README.md
├── cuDL
│   ├── __init__.py
│   ├── activations
│   │   ├── __init__.py
│   ├── initializers.py
│   ├── layers
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── dense.py
│   ├── loss.py
│   ├── metrics.py
│   ├── model.py
│   ├── optim
│   │   ├── __init__.py
│   ├── regularizers.py
│   └── utils.py
├── pytorch_train_fmnist_baseline.py  # a pytorch baseline useful for debugging our implementation
├── requirements.txt
└── train_fmnist.py                   # main file to train numpy neural networks
```
