import cv2
import random
import os, sys
from glob import glob
import numpy as np

from model import load_model,load_test_models
from model.core import train_model, valid_model
from model.GAN import Generator

from utils.data_loader import load_dataloader
from utils.config import ParserArguments,TestParserArguments
from utils.optim_utils import load_optimizer, load_loss_function, CosineWarmupLR

import torch
import torch.nn as nn
import warnings

import SimpleITK as sitk
from tqdm import tqdm
from sklearn.metrics import f1_score
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


warnings.filterwarnings(action='ignore')

torch.backends.cudnn.benchmark = True

# Seed
RANDOM_SEED = 1234
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

args = ParserArguments()

def image_padding(img_whole):
    img = np.zeros((400,400))
    img = img.astype(np.float32)
    
    img_whole = cv2.resize(img_whole, (400,img_whole.shape[0]))
    h,w = img_whole.shape
    #print(img_whole.shape)

    if h>400:
        img = img_whole
        pass
    elif (400 - h) != 0:
        gap = int((400 - h)/2)
        img[gap:gap+h,:] = img_whole
    elif (400 - w) != 0:
        gap = int((400 - w)/2)
        img[:,gap:gap+w] = img_whole
   

    return img

def center_crop(img):
	y, x = img.shape
	x_center = x/2.0
	
	x_min = int(x*0.2)
	x_max = int(2*x_center-x*0.2)
	
	
	img_cropped = img[0:y, x_min: x_max]
	
	
	return img_cropped

def random_crop(img, width, height):
	x = random.randint(0, img.shape[1] - width)
	y = random.randint(0, img.shape[0] - height)

	img_cropped = img[y:y+height, x:x+width]

	return img_cropped

def image_minmax(img):
	img_minmax = ((img - np.min(img)) / (np.max(img) - np.min(img))).copy()
	img_minmax = (img_minmax * 255).astype(np.uint8)
		
	return img_minmax

# def _load_image_list():
# 	vf_img_list = sorted(glob(os.path.join(args.data_root, 'VF','*.jpg')))[:]
# 	nonvf_img_list = random.sample(sorted(glob(os.path.join(args.data_root, 'Non-VF', '*.jpg'))),len(vf_img_list)*1)
	
# 	target_img_list = []

# 	split_idx = int(len(vf_img_list)*.6)
# 	split_idx_val = int(len(vf_img_list)*.2)
# 	# target_img_list.extend(vf_img_list[-split_idx_val:]*1)
# 	target_img_list.extend(vf_img_list[split_idx:split_idx+split_idx_val]*1)
# 	# target_img_list.extend(vf_img_list[:])
# 	vf_len = len(target_img_list)

# 	split_idx = int(len(nonvf_img_list)*.6)
# 	split_idx_val = int(len(nonvf_img_list)*.2)
# 	# target_img_list.extend(nonvf_img_list[-split_idx_val:]*1)
# 	target_img_list.extend(nonvf_img_list[split_idx:split_idx+split_idx_val]*1)
# 	# target_img_list.extend(nonvf_img_list[:])
# 	nonvf_len = len(target_img_list) -vf_len

# 	target_label_list = ([1] * vf_len)+([0] * nonvf_len)

# 	return target_img_list,target_label_list


def _load_image_list():
	vf_img_list = sorted(glob(os.path.join(args.data_root, 'VF','*.jpg')))[145:]
	nonvf_img_list = random.sample(sorted(glob(os.path.join(args.data_root, 'Non-VF', '*.jpg'))),len(vf_img_list)*4)
	
	target_img_list = []

	split_idx = int(len(vf_img_list))
	split_idx_val = int(len(vf_img_list)*.2)
	# target_img_list.extend(vf_img_list[-split_idx_val:]*1)
	target_img_list.extend(vf_img_list[0:split_idx])
	# target_img_list.extend(vf_img_list[:])
	vf_len = len(target_img_list)

	split_idx = int(len(nonvf_img_list))
	split_idx_val = int(len(nonvf_img_list)*.2)
	# target_img_list.extend(nonvf_img_list[-split_idx_val:]*1)
	target_img_list.extend(nonvf_img_list[0:split_idx])
	# target_img_list.extend(nonvf_img_list[:])
	nonvf_len = len(target_img_list)

	target_label_list = ([1] * vf_len)+([0] * nonvf_len)

	return target_img_list,target_label_list



if __name__ == '__main__':
	# Argument
	args = TestParserArguments()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# Load pre-trained model of GAN
	# g_path = '/home/vfuser/sungjoo/prediction_crop/pth/GAN/generator-291.pkl'
	# generator = Generator(1)
	# generator.load_state_dict(torch.load(g_path))
	# generator.eval()
	# print("GAN_model loaded")

	# Model
	model = load_test_models(args)

	print('Test start ...')
	
	target_img_list, target_label_list = _load_image_list()
	
	true_labels = []
	pred_labels = []
	prob_labels = []
	index = 0
	for img_path,label in tqdm(zip(target_img_list,target_label_list)):

		image = cv2.imread(img_path,0)
		
		# image = image.astype(np.float32) / image.max()
		image = image.astype(np.float32)
		
		image = center_crop(image)
		
		
		# image = image_padding(image)
		image = cv2.resize(image,(512,1024))
		
		
		image = np.stack((image,)*3,axis=0)
		image = image[None,...]
		
		index+=1
		image = torch.from_numpy(image)
		# GAN 
		# image = generator(image).detach()
		#gradcam 
	
		target_layer = model.module._blocks[-1]
		img = cv2.imread(img_path,0)
		img = center_crop(img)	
		img = np.float32(cv2.resize(img, (512,1024)))
		cv2.imwrite('/home/vfuser/sungjoo/prediction_crop/pytorch_grad_cam/original/'+str(index)+'.jpg',img)
		img_s = np.stack((img,)*3,axis=0)
		img_t = img_s[None,...]
		input_tensor = torch.from_numpy(img_t)
		
		cam = GradCAM(model=model, target_layer=target_layer, use_cuda=args.use_gpu)
		target_category =None
		grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
		grayscale_cam = grayscale_cam[0, :]
		
		color_image = cv2.imread(img_path,0)
		color_image= color_image.astype(np.float32)
		color_image = center_crop(color_image)	
		color_image = np.float32(cv2.resize(color_image,(512,1024)))/255
		
		color_image = np.stack((color_image,)*3,axis=2)
		

		visualization = show_cam_on_image(color_image, grayscale_cam)

		cv2.imwrite('/home/vfuser/sungjoo/prediction_crop/pytorch_grad_cam/result/'+str(index)+'.jpg',visualization)

		# GAN 
		# image = generator(image).detach()

		#concatnation
		label = torch.Tensor([label])
		print(image.shape,label.shape)
		image, label = image.to(device), label.to(device)

		pred_val = model(image)
		print(pred_val)
		_, pred_cls_val = torch.max(pred_val, 1)

		true_labels.extend(list(label.cpu().numpy().astype(int)))
		pred_labels.extend(list(pred_cls_val.cpu().numpy().astype(int)))
		# prob_labels.extend(torch.sigmoid(pred_val)[:,1].detach().cpu().numpy())
		prob_labels.extend(nn.functional.softmax(pred_val,dim=1)[:,1].detach().cpu().numpy())

	print(true_labels)
	print(pred_labels)
	print(prob_labels)

	new_lable = []
	for p in prob_labels:
		if p > 0.6:
			new_lable.append(1)
		else:
			new_lable.append(0)

	# validation performance
	class_f1 = f1_score(true_labels, pred_labels, average='weighted')
	print("Test f1 score: %.4f" %(class_f1))

	roc_auc = metrics.roc_auc_score(true_labels, pred_labels)
	print("Test AUC score: %.4f" %(roc_auc))
	
	print(classification_report(true_labels, pred_labels))

	print("-----------threshold:0.6-----------")
	print(true_labels)
	print(new_lable)
	# validation performance
	class_f1 = f1_score(true_labels, new_lable, average='weighted')
	print("Test f1 score: %.4f" %(class_f1))

	roc_auc = metrics.roc_auc_score(true_labels, new_lable)
	print("Test AUC score: %.4f" %(roc_auc))   
	print(classification_report(true_labels, new_lable))

	
