import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet18, resnet34, resnet50

def load_resnet(resnet_type, pretrained=True):
    if resnet_type == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif resnet_type == 'resnet34':
        return resnet34(pretrained=pretrained)
    elif resnet_type == 'resnet50':
        return resnet50(pretrained=pretrained)
    else:
        raise ValueError("resnet18, resnet34, and resnet50 are supported now.")

def load_model(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #####   Model   #####
    if 'efficientnet' in args.network:
        # model = EfficientNet.from_name(args.network)
        model = EfficientNet.from_pretrained(args.network)
        model._fc = nn.Linear(model._fc.in_features, args.num_classes)

    elif 'resnet' in args.network:
        model = load_resnet(args.network, pretrained=True)
        model.fc = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(model.fc.in_features, args.num_classes)
        )
        
    else:
        raise ValueError("resnet and efficientnet are only supported now.")
        # GPU settings
    if args.use_gpu:
        model.to(device)
        model = torch.nn.DataParallel(model)

    if args.resume:
        model.load_state_dict(torch.load(args.resume))

    return model

def load_test_models(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #####   Model   #####
    if 'efficientnet' in args.network:
        model = EfficientNet.from_name(args.network)
        model._fc = nn.Linear(model._fc.in_features, args.num_classes)
        print('ok')

    elif 'resnet' in args.network:
        model = load_resnet(args.network, pretrained=True)
        model.fc = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(model.fc.in_features, args.num_classes)
        )


    else:
        raise ValueError("resnet and efficientnet are only supported now.")

    # GPU settings
    if args.use_gpu:
        model.to(device)
        model = torch.nn.DataParallel(model)
    # try:
    model.load_state_dict(torch.load('/data/workspace/vfuser/sungjoo/Multi_clf/exp/stargan/epoch_099_val_loss_0.0335_val_f1_0.9917.pth'))
    print("Weight pth loaded: ",args.resume)
    # except:
    #     print('error')
    
    model.eval()

    return model