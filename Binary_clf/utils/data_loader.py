import os
import cv2
import random
import numpy as np
from glob import glob
from imgaug import augmenters as iaa
import pandas as pd

from model.GAN import Generator

import SimpleITK as sitk

import pickle
import gzip

import torch
from torch.utils.data import Dataset, DataLoader

random.seed(12356)


# def center_crop(img):
# 	y, x = img.shape
# 	x_center = x/2.0
	
# 	x_min = int(x*0.2)
# 	x_max = int(2*x_center-x*0.2)
	
	
# 	img_cropped = img[0:y, x_min: x_max]
# 	black = np.zeros((y,x))
# 	black = black.astype(np.float32)
# 	black[:,x_min: x_max] =img_cropped
	
# 	return black

def center_crop(img):
	y, x = img.shape
	x_center = x/2.0
	
	x_min = int(x*0.2)
	x_max = int(2*x_center-x*0.2)
	
	
	img_cropped = img[0:y, x_min: x_max]
	# print(img_cropped.shape)
	return img_cropped

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

# def image_padding(img_whole,width):
    
# 	img = np.zeros((2140,1760))
# 	img = img.astype(np.float32)
	
# 	h,w = img_whole.shape
# 	#print(img_whole.shape)
# 	img[width,w-width] =img_whole
#     return img

def random_crop(img, width, height):
	x = random.randint(0, img.shape[1] - width)
	y = random.randint(0, img.shape[0] - height)

	img_cropped = img[y:y+height, x:x+width]

	return img_cropped

def image_minmax(img):

	img = img.astype(np.float32)
	img_minmax = ((img - np.min(img)) / (np.max(img) - np.min(img))).copy()
	# img_minmax = (img_minmax * 255).astype(np.uint8)
		
	return img_minmax
	
def image_windowing(img, w_min=50, w_max=180):
    img_w = img.copy()

    img_w[img_w < w_min] = w_min
    img_w[img_w > w_max] = w_max

    return img_w

class VFDataset(Dataset):
	def __init__(self, is_Train, args):
		self.args = args
		self.is_Train = is_Train

		# g_path = os.path.join(self.args.gan_pth)
		# print(self.args.data_root)
		# # Load pre-trained model of GAN
		# self.generator = Generator(1)
		# self.generator.load_state_dict(torch.load(g_path))
		# self.generator.eval()
		# print("GAN_model loaded")


		self.img_list,self.label_list = self._load_image_list()

		print("# of %s images : %d" % ('training' if is_Train else 'validation', len(self.img_list)))

	def __getitem__(self, index):
		img_path = self.img_list[index]

		# Load Image
		
		image =  cv2.imread(img_path,0)
		
		
		# image = image.astype(np.float32) / image.max()
		image = image.astype(np.float32)
		
		# image = center_crop(image)
		# image = image_padding(image)

		image = cv2.resize(image,(256,400))
		
		# cv2.imwrite('/home/vfuser/sungjoo/prediction_crop/image_view/'+str(index)+'.jpg',image)
		
		image = np.stack((image,)*3,axis=0)
		# image = image[None,...]
		
		image = torch.from_numpy(image)

		# print("before GAN:", image.shape)
		# self.generator.eval()
		# image = self.generator(image)[0].detach().numpy()
		# print("after GAN:", image.shape)
		
		
		# Load Class
		class_idx = self.label_list[index]
		if self.is_Train:
			image = self.augment_img(image)

		return image, class_idx

	def __len__(self):
		return len(self.img_list)


	def _load_image_list(self):

		vf_img_list = sorted(glob(os.path.join(self.args.data_root, 'original','*.jpg')))
		nonvf_img_list = sorted(glob(os.path.join(self.args.data_root, 'results', '*.jpg')))
		print("VF:%s, None-VF:%s"%(len(vf_img_list),len(nonvf_img_list)))
		target_img_list = []
		# split_idx = int(len(vf_img_list)*self.args.train_val_ratio)
		split_idx = int(len(vf_img_list)*.9)
		split_idx_val = int(len(vf_img_list)*.1)
		target_img_list.extend(vf_img_list[:split_idx]) if self.is_Train else target_img_list.extend(vf_img_list[split_idx:]) 
		vf_len = len(target_img_list)

		# split_idx = int(len(nonvf_img_list)*self.args.train_val_ratio)
		split_idx = int(len(nonvf_img_list)*.9)
		split_idx_val = int(len(nonvf_img_list)*.1)
		target_img_list.extend(nonvf_img_list[:split_idx]) if self.is_Train else target_img_list.extend(nonvf_img_list[split_idx:]) 
		nonvf_len = len(target_img_list)
		target_label_list = ([1] * vf_len)+([0] * nonvf_len)

		return target_img_list,target_label_list


	def augment_img(self, img):
		if self.args.augmentation == 'heavy':
			scale_factor = random.uniform(1-self.args.scale_factor, 1+self.args.scale_factor)
			rot_factor = random.uniform(-self.args.rot_factor, self.args.rot_factor)

			seq = iaa.Sequential([
					iaa.Affine(
						scale=(scale_factor, scale_factor),
						rotate=rot_factor,
						translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
					),
					iaa.Sometimes(
						0.5,
						iaa.GaussianBlur(sigma=(0, 1.0))
					),
					iaa.Crop(percent=(0, 0.05)),
				], random_order=True)

		else:
			scale_factor = random.uniform(1-self.args.scale_factor, 1+self.args.scale_factor)
			rot_factor = random.uniform(-self.args.rot_factor, self.args.rot_factor)

			seq = iaa.Sequential([
					iaa.Affine(
						scale=(scale_factor, scale_factor),
						rotate=rot_factor
					)
				], random_order=True)

		seq_det = seq.to_deterministic()

		if np.ndim(img) == 2:
			# Grayscale
			img = seq_det.augment_images(img)
		# elif np.ndim(img) == 3:
		# 	# (C, H, W) -> (1, H, W, C)
		# 	img = np.moveaxis(img, 0, -1)[None, ...]
		# 	img = seq_det.augment_images(img)
		# 	img = np.moveaxis(img[0], -1, 0)

		return img

	def save_imglist(self):
		# save and compress.
		with gzip.open("exp/valid_list.pkl", 'wb') as f:
			pickle.dump(self.img_list, f)

		return

def load_dataloader(args):
	tr_set = VFDataset(is_Train=True, args=args)
	val_set = VFDataset(is_Train=False, args=args)

	batch_train = DataLoader(tr_set, batch_size=args.batch_size,shuffle=True, num_workers=28, pin_memory=True)
	batch_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=28, pin_memory=True)

	return batch_train, batch_val
