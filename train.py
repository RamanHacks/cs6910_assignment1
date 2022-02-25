from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import argparse
import wandb
import os
import cupy as cp
cp.random.seed(111)
import numpy as np
np.random.seed(111)
from tqdm import tqdm

from cuNN.optimizer import Momentum, SGD, AdaGrad, AdamW, RMSProp, Adam, Nadam
from cuNN.module import SimpleLinear
from cuNN.loss import CrossEntropy, MSE
from cuNN.utils import save, load, cutmix_batch, smooth_batch, mixup_batch


def setup_data(val_split, seed):
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', \
    'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Convert the labels to categorical one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    #Splitting Data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, \
        train_size = 1-val_split, test_size = val_split, random_state=seed)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def main(args):
    str_hidden_sizes = args.hidden_sizes
    args.hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
    # if only one number is passed, that means user wants to use the same number of hidden units in each layer
    if len(args.hidden_sizes) == 1:
        args.hidden_sizes = [args.hidden_sizes[0]] * (args.num_layers - 1)
    assert len(args.hidden_sizes) == args.num_layers - 1
    
    model_args = "_".join(
        [
            "HS_{}".format(str_hidden_sizes),
            "L_{}".format(args.num_layers),
            "O_{}".format(args.optimizer),
            "M_{}".format(args.momentum),
            "LO_{}".format(args.loss),
            "A_{}".format(args.activation),
            "LR_{}".format(args.learning_rate),
            "BS_{}".format(args.batch_size),
            "WI_{}".format(args.weight_init),
            "DR_{}".format(args.weight_decay_rate),
            "MU_{}".format(args.mixup_prob),
            "CM_{}".format(args.cutmix_prob),
            "SM_{}".format(args.smoothing_val),
            "E_{}".format(args.epochs),
            "S_{}".format(args.seed),
        ]
    )
    
    enable_wandb = args.enable_wandb
    if args.wandb_project is not None:
        # descriptive name for the run
        enable_wandb = True
        if args.wandb_entity is not None:
            wandb.init(
                project=args.wandb_project,
                name=model_args,
                config=vars(args),
                dir=args.model_dir,
                entity=args.wandb_entity,
            )
        else:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_suffix,
                config=vars(args),
                dir=args.model_dir,
            )
    else:
        if args.enable_wandb:
            wandb.init(config=args)
    
    model = SimpleLinear(784, 10, args.hidden_sizes, args.num_layers,
                args.activation, args.weight_init, args.regularizer, args.weight_decay_rate)

    if args.loss == "cross_entropy":
        criterion = CrossEntropy()
    elif args.loss == "mse":
        criterion = MSE()
    
    if args.optimizer == "sgd":
        opt = SGD(model.parameters, lr=args.learning_rate)
    elif args.optimizer == "momentum":
        opt = Momentum(model.parameters, lr=args.learning_rate, mu=args.momentum)
    elif args.optimizer == "adamw":
        opt = AdamW(model.parameters, lr=args.learning_rate)
    elif args.optimizer == "rmsprop":
        opt = RMSProp(model.parameters, lr=args.learning_rate, beta=args.momentum)
    elif args.optimizer == "adagrad":
        opt = AdaGrad(model.parameters, lr=args.learning_rate)
    elif args.optimizer == "adam":
        opt = Adam(model.parameters, lr=args.learning_rate)
    elif args.optimizer == "nadam":
        opt = Nadam(model.parameters, lr=args.learning_rate)
    
    tot_samples = len(x_train)
    batch_size = args.batch_size
    ep_acc, ep_loss = None, None 
    epoch_log = tqdm(total=args.epochs, desc='Epoch', position=0, unit='epoch', leave=True)
    step_no =0
    best_val_acc = 0
    
    (x_train_og, y_train), (x_val, y_val), (x_test, y_test) = setup_data(args.val_split, args.seed)
    x_train = x_train_og.copy()
    
    x_val = cp.asarray(x_val)
    x_test = cp.asarray(x_test)
    y_val = cp.asarray(y_val)
    y_test = cp.asarray(y_test)
    
    x_test = x_test.reshape(1,x_test.shape[0],-1).squeeze() / 255.0
    x_val = x_val.reshape(1,x_val.shape[0],-1).squeeze() / 255.0
    
    for epoch in range(args.epochs):
        if args.shuffle:
            index = np.random.permutation(tot_samples)
        else:
            index = np.arange(tot_samples)
        
        sum_loss, sum_acc = 0, 0
        for start_idx in range(0, tot_samples, batch_size):
            step_no +=1
            if start_idx + batch_size > tot_samples:
                end_idx = tot_samples
                len_t = tot_samples - start_idx
            else:
                end_idx = start_idx + batch_size 
                len_t = batch_size
                
            batch_index = index[start_idx:end_idx]
            x_batch = cp.asarray(np.array([x_train[i] for i in batch_index])) / 255.0
            y_batch = cp.asarray(np.array([y_train[i] for i in batch_index]))
            
            if args.smoothing_val > 0:
                y_batch = smooth_batch(y_batch, smoothing=args.smoothing_val)
            if args.epochs - epoch > args.epochs/2 and args.cutmix_prob > 0:
                x_batch, y_batch = cutmix_batch(x_batch, y_batch, prob=args.cutmix_prob)
            elif args.epochs - epoch > args.epochs/2 and args.mixup_prob > 0:
                x_batch, y_batch = mixup_batch(x_batch, y_batch, prob=args.mixup_prob)
                
            x_batch = x_val.reshape(1,len_t,-1).squeeze()
            
            y = model(x_batch)
            loss = criterion(y, y_batch)
            model.backward(loss)
            opt.apply()
            
            acc = cp.mean(y.argmax(axis=1) == y_batch.argmax(axis=1))
            sum_loss += loss['loss'] * len_t
            sum_acc += acc * len_t
        
    
        epoch_log.update(1)
        ep_loss = sum_loss/tot_samples
        ep_acc = sum_acc/tot_samples
        
        val_for = model.forward((x_val), True)
        val_loss = criterion(val_for, y_val)
        val_acc = cp.mean(val_for.argmax(axis=1) == y_val.argmax(axis=1))
        
        best_val_acc = max(val_acc, best_val_acc)
        
        epoch_log.set_postfix({'Train Loss': cp.round(ep_loss,3), 
                                'Val Loss': cp.round(val_loss['loss'],3), 
                                'Train Acc': cp.round(ep_acc,3), 
                                'Val Acc': cp.round(val_acc,3)})  
        if enable_wandb:    
            printing_dict = {
                    "epoch": epoch + 1,
                    "train_loss": float(cp.round(ep_loss, 3)),
                    "val_loss": float(cp.round(val_loss['loss'], 3)),
                    'train_acc': float(cp.round(ep_acc,3)), 
                    'val_acc': float(cp.round(val_acc,3)),
                    'best_val_acc': float(cp.round(best_val_acc,3)),
                }  
            wandb.log(printing_dict)
        
    # testing accuracy
    y_test_pred = model.forward((x_test), True)
    test_acc = cp.mean(y_test_pred.argmax(axis=1) == cp.array(y_test).argmax(axis=1))
    print('Test Accuracy:', test_acc)
    
    if args.model_dir is None:
        # set a descriptive model directory name
        args.model_dir = os.path.join(f"models/{model_args}",)
        os.makedirs(args.model_dir, exist_ok=True)
        
        save_path = os.path.join(args.model_dir, "ckpt.pkl")
        print("Saving models here: {}".format(save_path))
        save(model.parameters, save_path)
    
    if enable_wandb:
        wandb.log({"Test Accuracy": float(test_acc)})
    