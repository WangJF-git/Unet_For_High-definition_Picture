import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import *
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from data import *
from cut_patch import *
from measurement import *
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img, img_to_array
from matplotlib import pyplot
import glob
from keras import backend as K


class myUnet(object):
	def __init__(self, img_rows = 512, img_cols = 512, test_path="../test/test_pic", curve_path="../results/loss_curve",
				 test_mask_path="../test/test_mask",test_cut_path="../test/test_cut"):
		self.img_rows = img_rows
		self.img_cols = img_cols
		self.curve_path = curve_path
		self.test_path = test_path
		self.test_mask_path = test_mask_path
		self.test_cut_path = test_cut_path
# 参数初始化定义

	def load_train_data(self):
		mydata = dataProcess((self.img_rows, self.img_cols))
		imgs_train, imgs_mask_train = mydata.load_train_data()
		return imgs_train, imgs_mask_train
# 载入数据0

	def get_unet(self):
		inputs = Input((self.img_rows, self.img_cols,1))
		# 网络结构定义

		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		print ("conv1 shape:",conv1.shape)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		print ("conv1 shape:",conv1.shape)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		print ("pool1 shape:",pool1.shape)

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		print ("conv2 shape:",conv2.shape)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		print ("conv2 shape:",conv2.shape)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		print ("pool2 shape:",pool2.shape)

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		print ("conv3 shape:",conv3.shape)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		print ("conv3 shape:",conv3.shape)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		print ("pool3 shape:",pool3.shape)

		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = concatenate([drop4, up6], axis = 3)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = concatenate([conv3, up7], axis = 3)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = concatenate([conv2, up8], axis = 3)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = concatenate([conv1, up9], axis = 3)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

		model = Model(inputs = inputs, outputs = conv10)
		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = [dsc, tp, tn])
		return model

# 如果需要修改输入的格式，那么可以从以下开始修改，上面的结构部分不需要修改
	def train_model(self, BS=1, EPOCH=2):
		print("loading data")
		imgs_train, imgs_mask_train = self.load_train_data()
		X_train, X_valid, y_train, y_valid = train_test_split(imgs_train, imgs_mask_train, test_size=0.2, random_state=0)
		print("loading data done")
		model = self.get_unet()
		model.load_weights('my_unet.hdf5')
		print("got unet")
		model_checkpoint = ModelCheckpoint('my_unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		# 创建两个相同参数的实例
		data_gen_args = dict(
			rotation_range=45, width_shift_range=0.1, height_shift_range=0.1,
			shear_range=0.1, zoom_range=0.2, brightness_range=[0.7, 1.3],
			horizontal_flip=True, fill_mode='reflect')
		image_datagen = ImageDataGenerator(**data_gen_args)
		mask_datagen = ImageDataGenerator(**data_gen_args)
		# 为 fit 和 flow 函数提供相同的种子和关键字参数
		SEED1 = 1
		SEED2 = 2
		X_datagen = image_datagen.flow(X_train, batch_size=BS, seed=SEED1)
		y_datagen = mask_datagen.flow(y_train, batch_size=BS, seed=SEED1)
		X_valid_datagen = image_datagen.flow(X_valid, batch_size=BS, seed=SEED2)
		y_valid_datagen = mask_datagen.flow(y_valid, batch_size=BS, seed=SEED2)
		history = model.fit_generator(zip(X_datagen, y_datagen), validation_data=zip(X_valid_datagen, y_valid_datagen),
									  epochs=EPOCH, steps_per_epoch=len(X_train) // BS, validation_steps=len(X_valid) // BS,
									  verbose=1, callbacks=[model_checkpoint])
		#history = model.fit(imgs_train, imgs_mask_train, batch_size=BS, validation_split=0.2,
		#				 epochs=EPOCH, verbose=1, callbacks=[model_checkpoint])
		pyplot.plot(history.epoch, history.history['loss'],
					history.epoch, history.history['dsc'],
					history.epoch, history.history['val_loss'],
					history.epoch, history.history['val_dsc'])
		pyplot.legend(('loss', 'dsc', 'val_loss', 'val_dsc'), loc='upper right')
		pyplot.savefig(self.curve_path + '/loss_pic.png')

	def test_model(self, BS=1):
		#分割测试图片->投入网络进行预测->预测结果合并
		print('Predict test data')
		if os.path.exists(self.test_cut_path):
			delAll(self.test_cut_path)
		os.mkdir(self.test_cut_path)
		mydatapro = dataProcess((self.img_rows, self.img_cols))
		tests = glob.glob(self.test_path + "/*." + "JPG")
		tests.sort()
		tests_mask = glob.glob(self.test_mask_path + "/*." + "png")
		tests_mask.sort()
		num = 1
		model = self.get_unet()
		model.load_weights('my_unet.hdf5')
		result = np.ndarray(len(tests), dtype=np.float32)
		for fp,fm in zip(tests, tests_mask):
			img = load_img(fp, grayscale=True)
			mask = load_img(fm, grayscale=True)
			height, width = mydatapro.cutTestData(img, num, cut_size=512)
			img_test = mydatapro.load_test_data(num)
			imgs_mask_test = model.predict(img_test, batch_size=BS, verbose=1)
			imgs_mask_test[imgs_mask_test > 0.5] = 1
			imgs_mask_test[imgs_mask_test <= 0.5] = 0
			pre_mask = mydatapro.merge_cut_test(imgs_mask_test, height, width, num)
			pre_mask = pre_mask.astype(np.float32)
			mask = img_to_array(mask)
			mask /= 255
			b = pre_mask*mask
			result[num-1] = sum(sum(pre_mask*mask))*2 / (sum(sum(pre_mask)) + sum(sum(mask)))
			num = num + 1
		print(result)
		print(sum(result) / len(tests))
#vscode相对路径从整个文件的根目录开始算，pycharm从代码所在位置算


if __name__ == '__main__':
	myunet = myUnet()
	#myunet.train_model(BS=1, EPOCH=2) #训练unet,不用训练模型就注释这句话
	myunet.test_model(BS=1)
