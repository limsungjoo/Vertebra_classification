import os
import cv2
import random
import numpy as np
from glob import glob
from imgaug import augmenters as iaa
import os, numpy as np
import torch
import torch.utils.data as data


import pickle
import gzip

import torch
from torch.utils.data import Dataset, DataLoader

random.seed(12356)

class MultiDataLoader(Dataset):
	
	def __init__(self,is_Train, args):
		self.args = args
		self.is_Train = is_Train
		self.img_list = self._load_image_list()
					
					# self.transform = transform
        # self.random_crops = random_crops
        # self.trainval = trainval
	def _load_image_list(self):
		classes_list = os.listdir('/data/workspace/vfuser/sungjoo/stargan-master/stargan-master/RaFD/train')
		img_list = []
		for cls in classes_list :
			tmp_list = sorted(glob('/data/workspace/vfuser/sungjoo/stargan-master/stargan-master/RaFD/train/' + cls +'/*.jpg'))
			img_list += tmp_list[:int(len(tmp_list)*0.9)]
		
		return img_list	
	def __len__(self):
		return len(self.img_list)
	

	def set_label(self,img_path):
		class_name = img_path.split('/')[-2]

		if class_name =='Dongkang':
			y = [1,0,0]
		elif class_name =='Fuji':
			y = [0,1,0]
		elif class_name =='GE':
			y = [0,0,1]

		return y

		

	def __getitem__(self, index):
		img_path = self.img_list[index]

		x = cv2.imread(img_path,0)
		x = x.astype(np.float32)
		x = cv2.resize(x,(256,400))
		x = np.stack((x,)*3,axis=0)
		# x = x[None,...]
		x = torch.from_numpy(x)
		y = self.set_label(img_path)
		y = np.array(y)
		y = torch.from_numpy(y)


		return x, y


class MultiValDataLoader(Dataset):
	
	def __init__(self,is_Train, args):
		self.args = args
		self.is_Train = is_Train
		self.img_list = self._load_image_list()
        # self.transform = transform
        # self.random_crops = random_crops
        # self.trainval = trainval
	def __len__(self):
		return len(self.img_list)    
	def set_label(self,img_path):
					
					
		class_name =img_path.split('/')[-2]
		
		if class_name =='Dongkang':
			y = [1,0,0]
		elif class_name =='Fuji':
			y = [0,1,0]
		elif class_name =='GE':
			y = [0,0,1]
		return y

	def _load_image_list(self):
		classes_list = os.listdir('/data/workspace/vfuser/sungjoo/stargan-master/stargan-master/RaFD/train')
		img_list = []
		for cls in classes_list :
			tmp_list = sorted(glob('/data/workspace/vfuser/sungjoo/stargan-master/stargan-master/RaFD/train/' + cls +'/*.jpg'))
			img_list += tmp_list[int(len(tmp_list)*0.9):]
		return img_list	

	def __getitem__(self, index):
		img_path = self.img_list[index]
		
		x = cv2.imread(img_path,0)
		x = x.astype(np.float32)
		x = cv2.resize(x,(256,400))
		x = np.stack((x,)*3,axis=0)
		# x = x[None,...]
		x = torch.from_numpy(x)
		y = self.set_label(img_path)
		y = np.array(y)
		y = torch.from_numpy(y)
		return x, y

	


def load_dataloader(args):
	tr_set = MultiDataLoader(is_Train=True, args=args)

	val_set = MultiValDataLoader(is_Train=False, args=args)

	batch_train = DataLoader(tr_set, batch_size=args.batch_size,shuffle=True, num_workers=0, pin_memory=True)
	batch_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

	return batch_train, batch_val    
    

# class VFDataset(Dataset):
# 	def __init__(self, is_Train, args):
# 		self.args = args
# 		self.is_Train = is_Train

# 		# g_path = os.path.join(self.args.gan_pth)
# 		# print(self.args.data_root)
# 		# # Load pre-trained model of GAN
# 		# self.generator = Generator(1)
# 		# self.generator.load_state_dict(torch.load(g_path))
# 		# self.generator.eval()
# 		# print("GAN_model loaded")


# 		self.img_list,self.label_list = self._load_image_list()

# 		print("# of %s images : %d" % ('training' if is_Train else 'validation', len(self.img_list)))

# 	def __getitem__(self, index):
# 		img_path = self.img_list[index]

# 		# Load Image
		
# 		image =  cv2.imread(img_path,0)
		
		
# 		# image = image.astype(np.float32) / image.max()
# 		image = image.astype(np.float32)
		
# 		image = center_crop(image)
# 		# image = image_padding(image)

# 		image = cv2.resize(image,(512,1024))
		
# 		# cv2.imwrite('/home/vfuser/sungjoo/prediction_crop/image_view/'+str(index)+'.jpg',image)
		
# 		image = np.stack((image,)*3,axis=0)
# 		# image = image[None,...]
		
# 		image = torch.from_numpy(image)

# 		# print("before GAN:", image.shape)
# 		# self.generator.eval()
# 		# image = self.generator(image)[0].detach().numpy()
# 		# print("after GAN:", image.shape)
		
		
# 		# Load Class
# 		class_idx = self.label_list[index]
# 		if self.is_Train:
# 			image = self.augment_img(image)

# 		return image, class_idx

# 	def __len__(self):
# 		return len(self.img_list)


# 	def _load_image_list(self):
		
# 		vf_img_list = sorted(glob(os.path.join(self.args.data_root, 'VF','*.jpg')))[:145]
# 		nonvf_img_list = random.sample(sorted(glob(os.path.join(self.args.data_root, 'Non-VF', '*.jpg'))),len(vf_img_list)*5)
# 		print("VF:%s, None-VF:%s"%(len(vf_img_list),len(nonvf_img_list)))
# 		target_img_list = []
# 		# split_idx = int(len(vf_img_list)*self.args.train_val_ratio)
# 		split_idx = int(len(vf_img_list)*.9)
# 		split_idx_val = int(len(vf_img_list)*.1)
# 		target_img_list.extend(vf_img_list[:split_idx]*5) if self.is_Train else target_img_list.extend(vf_img_list[0:split_idx_val]*6) 
# 		vf_len = len(target_img_list)

# 		# split_idx = int(len(nonvf_img_list)*self.args.train_val_ratio)
# 		split_idx = int(len(nonvf_img_list)*.9)
# 		split_idx_val = int(len(nonvf_img_list)*.1)
# 		target_img_list.extend(nonvf_img_list[:split_idx]*5) if self.is_Train else target_img_list.extend(nonvf_img_list[0:split_idx_val]*6) 
# 		nonvf_len = len(target_img_list)
# 		target_label_list = ([1] * vf_len)+([0] * nonvf_len)

# 		return target_img_list,target_label_list


# 	def augment_img(self, img):
# 		if self.args.augmentation == 'heavy':
# 			scale_factor = random.uniform(1-self.args.scale_factor, 1+self.args.scale_factor)
# 			rot_factor = random.uniform(-self.args.rot_factor, self.args.rot_factor)

# 			seq = iaa.Sequential([
# 					iaa.Affine(
# 						scale=(scale_factor, scale_factor),
# 						rotate=rot_factor,
# 						translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
# 					),
# 					iaa.Sometimes(
# 						0.5,
# 						iaa.GaussianBlur(sigma=(0, 1.0))
# 					),
# 					iaa.Crop(percent=(0, 0.05)),
# 				], random_order=True)

# 		else:
# 			scale_factor = random.uniform(1-self.args.scale_factor, 1+self.args.scale_factor)
# 			rot_factor = random.uniform(-self.args.rot_factor, self.args.rot_factor)

# 			seq = iaa.Sequential([
# 					iaa.Affine(
# 						scale=(scale_factor, scale_factor),
# 						rotate=rot_factor
# 					)
# 				], random_order=True)

# 		seq_det = seq.to_deterministic()

# 		if np.ndim(img) == 2:
# 			# Grayscale
# 			img = seq_det.augment_images(img)
# 		# elif np.ndim(img) == 3:
# 		# 	# (C, H, W) -> (1, H, W, C)
# 		# 	img = np.moveaxis(img, 0, -1)[None, ...]
# 		# 	img = seq_det.augment_images(img)
# 		# 	img = np.moveaxis(img[0], -1, 0)

# 		return img

# 	def save_imglist(self):
# 		# save and compress.
# 		with gzip.open("exp/valid_list.pkl", 'wb') as f:
# 			pickle.dump(self.img_list, f)

# 		return


