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
from cuNN.utils import save, load, cutmix_batch, smooth_batch, mixup_batch, augmix_batch, cosine_decay_with_warmup


def setup_data(val_split, seed):
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
        [   "fmnist",
            "O_{}".format(args.optimizer),
            "A_{}".format(args.activation),
            "LR_{}".format(args.learning_rate),
            "L_{}".format(args.num_layers),
            "HS_{}".format(str_hidden_sizes),
            "BS_{}".format(args.batch_size),
            "WI_{}".format(args.weight_init),
            "DR_{}".format(args.weight_decay_rate),
            "loss_{}".format(args.loss),
            "M_{}".format(args.momentum),
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
    
    if args.augmix_prob > 0:
        model = SimpleLinear(784*3, 10, args.hidden_sizes, args.num_layers,
                    args.activation, args.weight_init, args.regularizer, args.weight_decay_rate)
    else:
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
    
    (x_train_og, y_train), (x_val, y_val), (x_test, y_test) = setup_data(args.val_split, args.seed)
    x_train = x_train_og.copy()
    
    x_val = cp.asarray(x_val)
    x_test = cp.asarray(x_test)
    y_val = cp.asarray(y_val)
    y_test = cp.asarray(y_test)
    
    if args.augmix_prob > 0:
        x_val = x_val[:, :, :, np.newaxis]
        x_val = np.repeat(x_val, 3, -1) 
        x_test = x_test[:, :, :, np.newaxis]
        x_test = np.repeat(x_test, 3, -1) 
        
    x_test = x_test.reshape(1,x_test.shape[0],-1).squeeze() / 255.0
    x_val = x_val.reshape(1,x_val.shape[0],-1).squeeze() / 255.0

    tot_samples = len(x_train)
    batch_size = args.batch_size
    ep_acc, ep_loss = None, None 
    epoch_log = tqdm(total=args.epochs, desc='Epoch', position=0, unit='epoch', leave=True)
    step_no =0
    total_steps = (int(tot_samples/batch_size) + 1)*args.epochs
    best_val_acc = 0
    for epoch in range(args.epochs):
        if args.shuffle:
            index = np.random.permutation(tot_samples)
        else:
            index = np.arange(tot_samples)
            
        if args.augmix_prob > 0:
            x_train = augmix_batch(x_train_og, prob=args.augmix_prob)
            
        max_lr = 0
        sum_loss, sum_acc = 0, 0
        for start_idx in range(0, tot_samples, batch_size):
            step_no +=1
            if start_idx + batch_size > tot_samples:
                end_idx = tot_samples
            else:
                end_idx = start_idx + batch_size 
            len_t = end_idx - start_idx
            
            # if len_t < batch_size/2:
            #     continue
            
            batch_index = index[start_idx:end_idx]
            x_batch = cp.asarray(np.array([x_train[i] for i in batch_index])) / 255.0
            y_batch = cp.asarray(np.array([y_train[i] for i in batch_index]))
            
            if args.smoothing_val > 0:
                y_batch = smooth_batch(y_batch, smoothing=args.smoothing_val)
            if args.epochs - epoch > args.epochs/2 and args.cutmix_prob > 0:
                x_batch, y_batch = cutmix_batch(x_batch, y_batch, prob=args.cutmix_prob)
            if args.epochs - epoch > args.epochs/2 and args.mixup_prob > 0:
                x_batch, y_batch = mixup_batch(x_batch, y_batch, prob=args.mixup_prob)
                
            x_batch = x_batch.reshape(1,len_t,-1).squeeze()
            y = model(x_batch)
            loss = criterion(y, y_batch)
            model.backward(loss)
            
            if args.use_lr_scheduler:
                cosine_lr = cosine_decay_with_warmup(
                    global_step = step_no,
                    learning_rate_base = args.learning_rate,
                    total_steps = total_steps,
                    warmup_steps = 0.2 * total_steps,
                    hold_base_rate_steps = 0.1 * total_steps,
                )
                
                # max_lr = max(max_lr, float(cosine_lr))
                opt.apply(lr=cp.asarray(cosine_lr))
            else:
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
                                'Train Acc': cp.round(ep_acc*100,4), 
                                'Val Acc': cp.round(val_acc*100,4)})
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
    epoch_log.close()
    
    # testing accuracy
    y_test_pred = model.forward((x_test), True)
    top_pred_ids = y_test_pred.argmax(axis=1)
    ground_truth_ids = cp.array(y_test).argmax(axis=1)
    test_acc = cp.mean(top_pred_ids == ground_truth_ids)
    
    print('Test Accuracy:', test_acc)
    if enable_wandb:
        wandb.log({"Test Accuracy": float(test_acc)})
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', \
        'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        wandb.log({"Confusion Matrix" : wandb.plot.confusion_matrix( 
            preds=list(cp.asnumpy(top_pred_ids)), y_true=list(cp.asnumpy(ground_truth_ids)),
            class_names=classes)})
    
    if args.model_dir is None:
        # set a descriptive model directory name
        args.model_dir = os.path.join(f"models/{model_args}",)
        os.makedirs(args.model_dir, exist_ok=True)
        
        save_path = os.path.join(args.model_dir, "ckpt.pkl")
        print("Saving models here: {}".format(save_path))
        save(model.parameters, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on the Fashion MNIST dataset"
    )
    parser.add_argument(
        "--model_dir", type=str, default=None, help="Directory to store the models",
    )
    # training arguments
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split for training data",
    )
    parser.add_argument(
        "--seed", type=int, default=133, help="Random seed for training",
    )
    parser.add_argument(
        "--shuffle", type=bool, default=True, help="Shuffle the training data",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for training",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of layers in the model",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=str,
        default="64",
        help="this is an integer or a comma-separated list of integers indicating the number of hidden units in each layer of the model.",
    )
    parser.add_argument(
        "--regularizer",
        type=str,
        default='l2',
        help="Regularizer (l1 or l2 or None) to choose for training",
    )
    parser.add_argument(
        "--weight_decay_rate",
        type=float,
        default=0,
        help="Regularizer lambda value",
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd", help="Optimizer to use for training",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD/rmsprop optimizer",
    )

    parser.add_argument(
        "--weight_init",
        type=str,
        default="xavier",
        help="Weight initialization scheme, can be one of 'xavier', 'he', 'uniform', 'random', 'zeros'",
    )
    parser.add_argument(
        "--activation", type=str, default="relu", 
        help="Activation function to use, can be one of 'relu', 'sigmoid', 'tanh'",
    )
    parser.add_argument(
        "--loss", type=str, default="cross_entropy", help="Loss function to use",
    )
    parser.add_argument(
        "--smoothing_val", type=float, default=0, help="Smoothing value"
    )
    parser.add_argument(
        "--cutmix_prob", type=float, default=0, help="Cutmix Prob"
    )
    parser.add_argument(
        "--mixup_prob", type=float, default=0, help="Mixup Prob"
    )
    
    parser.add_argument(
        "--augmix_prob", type=float, default=0., help="Augmix Prob"
    )
    parser.add_argument(
        "--use_lr_scheduler",
        action="store_true",
        default=False,
        help="Whether to use lr scheduler",
    )
    
    # add wandb arguments
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="W&B project name"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="W&B entity name"
    )
    parser.add_argument(
        "--enable_wandb",
        action="store_true",
        default=False,
        help="Whether to enable wandb",
    )
    args = parser.parse_args()
    
    main(args)