# Training neural networks

## Install Libraries
Install the correct version of `cupy` library based on your CUDA version from [here](https://docs.cupy.dev/en/stable/install.html).  
*Note: Cupy works only on NVIDIA GPUs.*

Install other requirements:
```
pip install -r requirements.txt
```

## Usage

```bash
python .\train_fmnist.py -h
usage: train_fmnist.py [-h] 
                [--model_dir MODEL_DIR] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--val_split VAL_SPLIT] [--seed SEED] [--shuffle SHUFFLE] [--learning_rate LEARNING_RATE] [--num_layers NUM_LAYERS]
                [--hidden_sizes HIDDEN_SIZES] [--regularizer REGULARIZER] [--weight_decay_rate WEIGHT_DECAY_RATE] [--optimizer OPTIMIZER] [--momentum MOMENTUM] [--weight_init WEIGHT_INIT]
                [--activation ACTIVATION] [--loss LOSS] [--smoothing_val SMOOTHING_VAL] [--cutmix_prob CUTMIX_PROB] [--mixup_prob MIXUP_PROB] [--augmix_prob AUGMIX_PROB] [--use_lr_scheduler]
                [--wandb_project WANDB_PROJECT] [--wandb_entity WANDB_ENTITY] [--enable_wandb]

Train a model on the Fashion MNIST dataset

options:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        Directory to store the models
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
                        this is an integer or a comma-separated list of integers indicating the number of hidden units in each layer of the model.
  --regularizer REGULARIZER
                        Regularizer (l1 or l2 or None) to choose for training
  --weight_decay_rate WEIGHT_DECAY_RATE
                        Regularizer lambda value
  --optimizer OPTIMIZER
                        Optimizer to use for training
  --momentum MOMENTUM   Momentum for SGD/rmsprop optimizer
  --weight_init WEIGHT_INIT
                        Weight initialization scheme, can be one of 'xavier', 'he', 'uniform', 'random', 'zeros'
  --activation ACTIVATION
                        Activation function to use, can be one of 'relu', 'sigmoid', 'tanh'
  --loss LOSS           Loss function to use
  --smoothing_val SMOOTHING_VAL
                        Smoothing value
  --cutmix_prob CUTMIX_PROB
                        Cutmix Prob
  --mixup_prob MIXUP_PROB
                        Mixup Prob
  --augmix_prob AUGMIX_PROB
                        Augmix Prob
  --use_lr_scheduler    Whether to use lr scheduler
  --wandb_project WANDB_PROJECT
                        W&B project name
  --wandb_entity WANDB_ENTITY
                        W&B entity name
  --enable_wandb        Whether to enable wandb
```

Example command to train your network:
```
python train_fmnist.py --learning_rate 0.001 --activation=silu --batch_size=64 --hidden_sizes=128 --learning_rate=0.001 --num_layers=3 --optimizer=nadam --weight_decay_rate=0 --weight_init=xavier --wandb_entity=gowthamr --wandb_project=dl-course --use_lr_scheduler --smoothing_val 0.1 --augmix_prob 0.5 --epochs=30
```

Folder Structure:

```
├── cuNN
│   ├── activation.py
│   ├── initializer.py
│   ├── __init__.py
│   ├── loss.py
│   ├── module.py
│   ├── optimizer.py
│   └── utils.py
├── README.md
├── requirements.txt
├── sweep.yaml
└── train.py
```