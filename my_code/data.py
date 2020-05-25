from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
from cut_patch import *
import cv2


class dataProcess(object):
	def __init__(self, out_size=(512,512), input_pic="../my_pic/cut_pic", input_mask="../my_pic/cut_mask",
				 input_extra_pic="../my_pic/cut_e_pic", input_extra_mask="../my_pic/cut_e_mask",
				 test_path="../test/test_pic", test_cut_path="../test/test_cut", result_path="../results/pre_mask",
				 npy_path = "../npydata", img_type = "tif"):
		# 数据处理类，初始化
		self.out_size = out_size
		self.input_pic = input_pic
		self.input_mask = input_mask
		self.input_extra_pic = input_extra_pic
		self.input_extra_mask = input_extra_mask
		self.test_path = test_path
		self.npy_path = npy_path
		self.img_type = img_type
		self.trains = glob.glob(input_pic + "/*." + img_type)  # glob(r"\t")加r可以防止字符转义
		self.trains.sort()
		self.masks = glob.glob(input_mask + "/*." + img_type)  # glob(r"\t")加r可以防止字符转义
		self.masks.sort()
		self.e_trains = glob.glob(input_extra_pic + "/*." + img_type)  # glob(r"\t")加r可以防止字符转义
		self.e_trains.sort()
		self.e_masks = glob.glob(input_extra_mask + "/*." + img_type)  # glob(r"\t")加r可以防止字符转义
		self.e_masks.sort()
		self.test_cut_path = test_cut_path
		self.result_path = result_path

	# 创建训练数据
	def create_train_data(self):
		if os.path.exists(self.npy_path):
			delAll(self.npy_path)
		os.mkdir(self.npy_path)
		rows, cols = self.out_size[0], self.out_size[1]
		trains = self.trains
		masks = self.masks
		e_trains = self.e_trains
		e_masks = self.e_masks
		print('-' * 30)
		print('Creating training images...')
		print('-' * 30)
		imgdatas = np.ndarray((len(trains)+len(e_trains), rows, cols, 1), dtype=np.uint8)
		imglabels = np.ndarray((len(masks)+len(e_masks), rows, cols, 1), dtype=np.uint8)
		i = 0
		for f_train, f_mask in zip(trains, masks):
			img = load_img(f_train, grayscale=True)
			label = load_img(f_mask, grayscale=True)
			img = img_to_array(img)
			label = img_to_array(label)
			imgdatas[i] = img
			imglabels[i] = label
			if i % 10 == 0:
				print('Done: {0}/{1} images'.format(i, len(trains)+len(e_trains)))
			i += 1
		for f_train, f_mask in zip(e_trains, e_masks):
			img = load_img(f_train, grayscale=True)
			label = load_img(f_mask, grayscale=True)
			img = img_to_array(img)
			label = img_to_array(label)
			imgdatas[i] = img
			imglabels[i] = label
			if i % 10 == 0:
				print('Done: {0}/{1} images'.format(i, len(trains)+len(e_trains)))
			i += 1
		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
		print('Saving to train.npy files done.')

# 加载训练图片与mask
	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255
		mean = imgs_train.mean(axis=1)
		mean = mean.mean(axis=1)
		mean = mean.reshape((mean.shape[0], 1, 1, 1))
		imgs_train -= mean
		imgs_mask_train /= 255
		# 做一个阈值处理，输出的概率值大于0.5的就认为是对象，否则认为是背景
		imgs_mask_train[imgs_mask_train > 0.5] = 1
		imgs_mask_train[imgs_mask_train <= 0.5] = 0
		print('Loading train data done.')
		return imgs_train,imgs_mask_train

	def cutTestData(self, input, pre_index, cut_size=512):
		print('-' * 30)
		print('Cutting num.{} test images...'.format(pre_index))
		print('-' * 30)
		img_type = self.img_type
		save_dir = self.test_cut_path + "/" + str(pre_index).zfill(3)
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		arr = img_to_array(input)
		rows = arr.shape[0] // cut_size
		cols = arr.shape[1] // cut_size
		for i in range(rows):
			for j in range(cols):
				TopRightX = i * cut_size
				TopRightY = j * cut_size
				arr_cut = arr[TopRightX:TopRightX + cut_size, TopRightY:TopRightY + cut_size, :]
				img_tmp = array_to_img(arr_cut)
				img_tmp.save(save_dir + "/" + str(i).zfill(2) + "_" + str(j).zfill(2) + "." + img_type)
			if cols * cut_size < arr.shape[1]:
				TopRightX = i * cut_size
				TopRightY = arr.shape[1] - cut_size
				arr_cut = arr[TopRightX:TopRightX + cut_size, TopRightY:TopRightY + cut_size, :]
				img_tmp = array_to_img(arr_cut)
				img_tmp.save(save_dir + "/" + str(i).zfill(2) + "_" + str(cols).zfill(2) + "." + img_type)
		if rows * cut_size < arr.shape[0]:
			TopRightX = arr.shape[0] - cut_size
			for j in range(cols):
				TopRightY = j * cut_size
				arr_cut = arr[TopRightX:TopRightX + cut_size, TopRightY:TopRightY + cut_size, :]
				img_tmp = array_to_img(arr_cut)
				img_tmp.save(save_dir + "/" + str(rows).zfill(2) + "_" + str(j).zfill(2) + "." + img_type)
			if cols * cut_size < arr.shape[1]:
				TopRightY = arr.shape[1] - cut_size
				arr_cut = arr[TopRightX:TopRightX + cut_size, TopRightY:TopRightY + cut_size, :]
				img_tmp = array_to_img(arr_cut)
				img_tmp.save(save_dir + "/" + str(rows).zfill(2) + "_" + str(cols).zfill(2) + "." + img_type)
		return arr.shape[0], arr.shape[1]

	# 创建测试数据
	def load_test_data(self, pre_index):
		i = 0
		rows, cols = self.out_size[0], self.out_size[1]
		print('-' * 30)
		print('Loading test images...')
		print('-' * 30)
		input_path = self.test_cut_path + "/" + str(pre_index).zfill(3)
		tests = glob.glob(input_path + "/*." + self.img_type)
		tests.sort()
		imgdatas = np.ndarray((len(tests), rows, cols, 1), dtype=np.uint8)
		for f_test in tests:
			img = load_img(f_test, grayscale=True)
			img = img_to_array(img)
			imgdatas[i] = img
			i += 1
		imgs_test = imgdatas.astype('float32')
		imgs_test /= 255
		mean = imgs_test.mean(axis=1)
		mean = mean.mean(axis=1)
		mean = mean.reshape((mean.shape[0], 1, 1, 1))
		imgs_test -= mean
		print('Loading test images done')
		return imgs_test

	# 合并测试数据
	def merge_cut_test(self, masks, height, width, pre_index):
		print('Merging test images...')
		size = self.out_size[0]
		save_dir = self.result_path
		merge_mask = np.ndarray((height, width, 1), dtype=np.float, )
		tl_x, tl_y = 0, 0
		index = 0
		while tl_x < height or tl_y < width:
			if tl_x + size > height:
				tl_x = height - size
			if tl_y + size > width:
				tl_y = width -size
			merge_mask[tl_x:tl_x+size, tl_y:tl_y+size] = masks[index]
			index += 1
			tl_y += size
			if tl_y == width:
				tl_x += size
				if tl_x != height:
					tl_y = 0
		img = array_to_img(merge_mask)
		img.save(save_dir + "/" + str(pre_index).zfill(3) + ".png")
		print('Finish merging images...')
		return merge_mask


if __name__ == "__main__":
	mydata = dataProcess((512, 512))
	mydata.create_train_data()    #从cut_pic和cut_mask中将数据转化为.npy格式

	imgs_train,imgs_mask_train = mydata.load_train_data()
	print(imgs_train.shape,imgs_mask_train.shape)
