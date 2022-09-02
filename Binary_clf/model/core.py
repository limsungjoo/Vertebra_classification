from sklearn.metrics import f1_score

from utils import AverageMeter
from utils.optim_utils import get_current_lr

import torch
import torch.nn as nn
import torch.nn.functional as F

def train_model(epoch, batch_train, device, optimizer, model, criterion, lr_fn, args):
    model.train()

    ## Training
    true_labels = []
    pred_labels = []
    train_loss = AverageMeter()
    for i, (x_tr, y_tr) in enumerate(batch_train):
        # Zero grad
        for p in model.parameters(): p.grad = None 

        x_tr, y_tr = x_tr.to(device), y_tr.to(device)
        print(y_tr)
        pred = model(x_tr)
        print(pred)
        loss = criterion(pred, y_tr)

        loss.backward()
        optimizer.step()

        # cosing annealing
        if lr_fn is not None:
            lr_fn.step(epoch * len(batch_train) + i)

        _, pred_cls = torch.max(pred, 1)

        train_loss.update(loss.item(), len(x_tr))
        true_labels.extend(list(y_tr.cpu().numpy().astype(int)))
        pred_labels.extend(list(pred_cls.cpu().numpy().astype(int)))

        print(">>> Epoch [%3d/%3d] | Iter [%3d/%3d] | Loss %.6f" % (epoch+1, args.nb_epoch, i+1, len(batch_train), train_loss.avg), end='\r')

    # train performance
    class_f1 = f1_score(true_labels, pred_labels, average='weighted')
    print("\n>>> Training F1 : %.4f (LR %.7f)" % (class_f1, get_current_lr(optimizer)))
    
    return train_loss.avg, class_f1


def valid_model(epoch, batch_val, device, model, criterion, args):
    model.eval()

    val_loss = AverageMeter()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for j, (x_val, y_val) in enumerate(batch_val):
            x_val, y_val = x_val.to(device), y_val.to(device)

            pred_val = model(x_val)
            loss_val = criterion(pred_val, y_val)

            _, pred_cls_val = torch.max(pred_val, 1)

            val_loss.update(loss_val.item(), len(x_val))
            true_labels.extend(list(y_val.cpu().numpy().astype(int)))
            pred_labels.extend(list(pred_cls_val.cpu().numpy().astype(int)))

    # validation performance
    class_f1 = f1_score(true_labels, pred_labels, average='weighted')
    print(">>> Validation F1 : %.4f" % class_f1)
    
    return val_loss.avg, class_f1