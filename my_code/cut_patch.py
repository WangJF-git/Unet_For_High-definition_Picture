from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import random
import cv2 as cv


class myCutPatches(object):
    """
    随机裁剪的类
    """
    def __init__(self, cut_size=512, img_path="../my_pic/raw_pic", mask_path="../my_pic/raw_mask",
                 cut_train_path="../my_pic/cut_pic", cut_label_path="../my_pic/cut_mask",
                 extra_train_path="../my_pic/extra_pic", extra_label_path="../my_pic/extra_mask",
                 pic_type="JPG", mask_type="png"):
        self.cut_size = cut_size
        self.img_path = img_path
        self.mask_path = mask_path
        self.cut_train_path = cut_train_path
        self.cut_label_path = cut_label_path
        self.extra_train_path = extra_train_path
        self.extra_label_path = extra_label_path
        self.pic_type = pic_type
        self.mask_type = mask_type
        self.raw_img = glob.glob(img_path+"/*."+pic_type) #glob(r"\t")加r可以防止字符转义
        self.raw_img.sort()
        self.mask_img = glob.glob(mask_path+"/*."+mask_type)
        self.mask_img.sort()

    def cutPatches(self, cutNum=20, bg_percent=0.2):
        #merge, save to merge_path
        img = self.raw_img
        mask = self.mask_img
        if os.path.exists(self.cut_train_path):
            delAll(self.cut_train_path)  ##清空上一次的随机裁剪
        if os.path.exists(self.cut_label_path):
            delAll(self.cut_label_path)
        os.mkdir(self.cut_train_path)
        os.mkdir(self.cut_label_path)
        if len(img) != len(mask) or len(img) == 0 or len(mask) == 0:
            print("trains can't match labels")
            return 0
        print('-'*30)
        print('Randomly cutting pictures and masks...')
        print('Number of raw image is', len(img))
        print('-' * 30)
        i = 0
        for f1, f2 in zip(img, mask):
            img_t = load_img(f1)  # 图像是灰度图，rgb三通道数值相同
            img_l = load_img(f2)
            x_t = img_to_array(img_t)
            x_l = img_to_array(img_l)
            self.randomCut(x_t, x_l, str(i).zfill(3), cutNum, bg_percent, self.cut_train_path, self.cut_label_path)
            i = i + 1
        print('Finish cutting')

    def randomCut(self, img_input, mask_input, prefix, cutNum, bg_percent, trainPath, labelPath, img_type="tif"):
        img = img_input
        mask = mask_input
        cut_size = self.cut_size
        bg_num = round(cutNum * bg_percent)
        tar_num = cutNum - bg_num
        i = 0
        while bg_num >= 0 or tar_num >= 0:
            #  生成剪切图像的左上角XY坐标
            TopRightX = random.randint(0, img.shape[0])
            TopRightY = random.randint(0, img.shape[1])
            #  设定防止裁剪图像超过原图像的边界的判断条件
            if TopRightY + cut_size <= img.shape[1] and TopRightX + cut_size <= img.shape[0]:
                img_cut = img[TopRightX:TopRightX + cut_size, TopRightY:TopRightY + cut_size, :]
                mask_cut = mask[TopRightX:TopRightX + cut_size, TopRightY:TopRightY + cut_size, :]
                img_tmp = array_to_img(img_cut)
                mask_tmp = array_to_img(mask_cut)
                data = np.asarray(mask_cut, dtype=int)
                if data.max() == 0:
                    bg_num -= 1
                    if bg_num < 0:
                        continue
                else:
                    tar_num -= 1
                    if tar_num < 0:
                        continue
                img_tmp.save(trainPath + "/" + prefix + "_" + str(i).zfill(3) + "." + img_type)
                mask_tmp.save(labelPath + "/" + prefix + "_" + str(i).zfill(3) + "." + img_type)
                i = i + 1

    def imgGenerator(self, Gauss=True, EquHist=True, Resize=True):
    # 原图数据增强，包括高斯模糊，直方图均衡化，缩小
        if os.path.exists(self.extra_train_path):
            delAll(self.extra_train_path)  ##清空上一次的数据增强
        if os.path.exists(self.extra_label_path):
            delAll(self.extra_label_path)
        os.mkdir(self.extra_train_path)
        os.mkdir(self.extra_label_path)

        img = self.raw_img
        mask = self.mask_img
        if len(img) != len(mask) or len(img) == 0 or len(mask) == 0:
            print("trains can't match labels")
            return 0
        print('-' * 30)
        print('Generating pictures and masks...')
        print('Number of raw image is', len(img))
        print('-' * 30)
        i = 0
        for f1, f2 in zip(img, mask):
            x = cv.imread(f1, cv.IMREAD_GRAYSCALE)
            y = cv.imread(f2, cv.IMREAD_GRAYSCALE)
            if Gauss:
                x_Gauss = cv.GaussianBlur(x, (17, 17), 0)
                cv.imwrite(self.extra_train_path + '/' + str(i).zfill(3) + '_0.JPG', x_Gauss)
                cv.imwrite(self.extra_label_path + '/' + str(i).zfill(3) + '_0.png', y)
            if EquHist:
                x_Equ = cv.equalizeHist(x)
                cv.imwrite(self.extra_train_path + '/' + str(i).zfill(3) + '_1.JPG', x_Equ)
                cv.imwrite(self.extra_label_path + '/' + str(i).zfill(3) + '_1.png', y)
            if Resize:
                x_res = cv.resize(x, (0, 0), fx=0.5, fy=0.5)
                y_res = cv.resize(y, (0, 0), fx=0.5, fy=0.5)
                cv.imwrite(self.extra_train_path + '/' + str(i).zfill(3) + '_2.JPG', x_res)
                cv.imwrite(self.extra_label_path + '/' + str(i).zfill(3) + '_2.png', y_res)
            i = i + 1
        print('Finish generator')

def delAll(path):
    if os.path.isdir(path):
        files = os.listdir(path)  # ['a.doc', 'b.xls', 'c.ppt']
        # 遍历并删除文件
        for file in files:
            p = os.path.join(path, file)
            if os.path.isdir(p):
                # 递归
                delAll(p)
            else:
                os.remove(p)
        # 删除文件夹
        os.rmdir(path)
    else:
        os.remove(path)


if __name__ == "__main__":
    cut = myCutPatches(cut_size=512) #裁剪原图像，放入cut_pic和cut_mask, (with_target)是否必须有目标
    cut.cutPatches(cutNum=0, bg_percent=0.2)             #为图像分割裁剪
    cut.imgGenerator(Gauss=False, EquHist=True, Resize=False) #图像增强
    e_cut = myCutPatches(cut_size=512, img_path="../my_pic/extra_pic", mask_path="../my_pic/extra_mask",
                 cut_train_path="../my_pic/cut_e_pic", cut_label_path="../my_pic/cut_e_mask")  #生成增强后的图像
    e_cut.cutPatches(cutNum=5, bg_percent=0.2)