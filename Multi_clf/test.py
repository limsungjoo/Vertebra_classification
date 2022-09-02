import cv2
import random
import os, sys
from glob import glob
import numpy as np
import pandas as pd

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


def _load_image_list():
	# img_excel=[]
	# img_list =[]
	# img_path = '/data/workspace/vfuser/VF/data/spine_data3/img_JPG'
	# excel_path = '/home/vfuser/sungjoo/data/vf_table.xlsx'
	# excel = pd.read_excel(excel_path)
	# for i in range(len(excel['fullname'].values)):
	# 	if excel['manufacturer'].values[i] =='FUJI PHOTO FILM ' or excel['manufacturer'].values[i] == 'GE Healthcare ' :
	# #         print(excel['fullname'].values[i])
	# 		if excel['pa_ap'].values[i] ==False:
	# #             print(excel['fullname'].values[i])
	# 			if excel['coments'].values[i] == 2 :
	# 				# print(excel['fullname'].values[i])
	# 				img = excel['fullname'].values[i]
	# 				img = img.split('.')[-2]
	# 				img_excel.append(img)
	# for image in os.listdir(img_path):
	# 	name = image.split('.')[-2]
	# 	for ex_image in img_excel:
	# 		if name == ex_image:
	# 			img_list.append(os.path.join(img_path,name+'.jpg'))
	img_list =glob('/data/workspace/vfuser/sungjoo/stargan-master/stargan-master/stargan_custom/test_image/results/*')

	return img_list



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
	
	target_img_list = _load_image_list()
	
	true_labels = []
	pred_labels = []
	prob_labels = []
	index = 0
	position =[]
	name=[]
	k=0
	j=0
	for img_path in tqdm(target_img_list):
		# print(img_path)
		image = cv2.imread(img_path,0)
		try:
			image = image.astype(np.float32)
			image = cv2.resize(image,(256,400))

			# label = set_label(img_path)
			# label = np.array(label)

			# cv2.imwrite('/home/vfuser/sungjoo/Multi_clf/image_test_view/'+str(index)+'.jpg',image)
			# index+=1
			image = np.stack((image,)*3,axis=0)
			image = image[None,...]
			image = torch.from_numpy(image)
			# print(image.shape)
			# GAN 
			# image = generator(image).detach()
			#gradcam 
		
			# target_layer = model.module._blocks[-1]
			# img = cv2.imread(img_path,0)
			# img = center_crop(img)	
			# img = np.float32(cv2.resize(img, (512,1024)))
			# cv2.imwrite('/home/vfuser/sungjoo/prediction_crop/pytorch_grad_cam/original/'+str(index)+'.jpg',img)
			# img_s = np.stack((img,)*3,axis=0)
			# img_t = img_s[None,...]
			# input_tensor = torch.from_numpy(img_t)
			
			# cam = GradCAM(model=model, target_layer=target_layer, use_cuda=args.use_gpu)
			# target_category =None
			# grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
			# grayscale_cam = grayscale_cam[0, :]
			
			# color_image = cv2.imread(img_path,0)
			# color_image= color_image.astype(np.float32)
			# color_image = center_crop(color_image)	
			# color_image = np.float32(cv2.resize(color_image,(512,1024)))/255
			
			# color_image = np.stack((color_image,)*3,axis=2)
			

			# visualization = show_cam_on_image(color_image, grayscale_cam)

			# cv2.imwrite('/home/vfuser/sungjoo/prediction_crop/pytorch_grad_cam/result/'+str(index)+'.jpg',visualization)

			# GAN 
			# image = generator(image).detach()

			#concatnation
			# label = torch.Tensor([label])

			# print(image.shape)
			# image,label = image.to(device), label.to(device)
			image = image.to(device)

			pred_val = model(image)
			
			_,pred_cls_val = torch.max(pred_val, 1)
			print('index'+str(index),pred_cls_val.cpu().numpy().astype(int))
			print(pred_cls_val.cpu().numpy().astype(int)[0])
			res = pred_cls_val.cpu().numpy().astype(int)[0]
			if res ==0:
				k +=1
			if res ==2:
				j+=1
			
		
			# print(img_path.split('/')[-1])
			name.append(img_path.split('/')[-1])
			position.append(pred_cls_val.cpu().numpy().astype(int)[0])

		except:
			print(img_path)
			break
	acc = j /(k+j)
	print(acc)
	# print(name)

	# dic = {"Image" : name, "position" : position}
	# df = pd.DataFrame(dic)
	# df.to_excel('/home/vfuser/sungjoo/Multi_clf/position.xlsx')




	# 	true_labels.extend(list(torch.argmax(label,axis=1).cpu().numpy().astype(int)))
	# 	pred_labels.extend(list(pred_cls_val.cpu().numpy().astype(int)))
	# 	# prob_labels.extend(torch.sigmoid(pred_val)[:,1].detach().cpu().numpy())
	# 	prob_labels.extend(nn.functional.softmax(pred_val,dim=1)[:,1].detach().cpu().numpy())

	# print(true_labels)
	# print(pred_labels)
	# print(prob_labels)

	# new_lable = []
	# for p in prob_labels:
	# 	if p > 0.6:
	# 		new_lable.append(1)
	# 	else:
	# 		new_lable.append(0)

	# # validation performance
	# class_f1 = f1_score(true_labels, pred_labels, average='weighted')
	# print("Test f1 score: %.4f" %(class_f1))

	# roc_auc = metrics.roc_auc_score(true_labels, pred_labels)
	# print("Test AUC score: %.4f" %(roc_auc))
	
	# print(classification_report(true_labels, pred_labels))

	# print("-----------threshold:0.6-----------")
	# print(true_labels)
	# print(new_lable)
	# # validation performance
	# class_f1 = f1_score(true_labels, new_lable, average='weighted')
	# print("Test f1 score: %.4f" %(class_f1))

	# roc_auc = metrics.roc_auc_score(true_labels, new_lable)
	# print("Test AUC score: %.4f" %(roc_auc))   
	# print(classification_report(true_labels, new_lable))

	
