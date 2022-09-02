import cv2
import random
import os, sys
from glob import glob
import numpy as np

from model import load_model
from model.core import train_model, valid_model

from utils.data_loader import load_dataloader
from utils.config import ParserArguments
from utils.optim_utils import load_optimizer, load_loss_function, CosineWarmupLR

import torch
import torch.nn as nn
import warnings


warnings.filterwarnings(action='ignore')

torch.backends.cudnn.benchmark = True

# Seed
RANDOM_SEED = 1234
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
index =0
if __name__ == '__main__':
    # Argument
    args = ParserArguments()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model
    model = load_model(args)

    # Loss
    criterion = load_loss_function(args).to(device)

    # Optimizer
    optimizer = load_optimizer(model, args)
    
    if args.mode == 'train':  ## for train mode
        print('Training start ...')
        train_loader, val_loader = load_dataloader(args)
        res = next(iter(train_loader))
        cv2.imwrite('/data/workspace/vfuser/sungjoo/Multi_clf/image_view/'+str(index)+'.jpg',res[0][0][0].numpy())
        # index +=1
        
        


        # Learning Rate Scheduler
        # lr_fn = CosineWarmupLR(optimizer=optimizer, epochs=args.nb_epoch, iter_in_one_epoch=len(train_loader), lr_min=args.min_lr,
        #                        warmup_epochs=args.warmup_epoch)
        lr_fn = None
        #####   Training and Validation loop   #####
        best_f1 = 0
        best_loss = 100
        for epoch in range(args.nb_epoch):
            train_loss, train_f1 = train_model(epoch, train_loader, device, optimizer, model, criterion, lr_fn, args)
            val_loss, val_f1 = valid_model(epoch, val_loader, device, model, criterion, args)

            # if val_f1 > best_f1:
            #     best_f1 = val_f1
            # # if train_loss < best_loss:
            # #     best_loss = train_loss
            #     # Remove previous best weights
            #     for previous_weights in glob(os.path.join(args.exp, '*.pth')):
            #         os.remove(previous_weights)

            torch.save(model.state_dict(),
                        os.path.join(args.exp, 'epoch_%03d_val_loss_%.4f_val_f1_%.4f.pth'%(epoch, val_loss, val_f1)))
            
                # print("\r>>> Best score updated : F1 %.4f in %d epoch.\n" % (val_f1, epoch+1))
            # else:
            #     print("\n")
        # torch.save(model.state_dict(),
        #                     os.path.join(args.exp, 'epoch_%03d_val_loss_%.4f_val_f1_%.4f.pth'%(epoch, val_loss, val_f1)))

 
        