import cv2 as cv
import os
import numpy as np
import glob
from cut_patch import delAll

DEBUG = True

class myResProcess(object):
    def __init__(self, result_path="../results/pre_mask", final_path="../results/fianl_mask",
                 pic_type="png"):
        self.result_path = result_path
        self.final_path = final_path
        self.pic_type = pic_type
        self.result_img = glob.glob(result_path+"/*."+pic_type) #glob(r"\t")加r可以防止字符转义
        self.result_img.sort()

    def resProcess(self, MIN_AREA=10000, ITER=10):
        data = self.result_img
        index = 1
        if os.path.exists(self.final_path):
            delAll(self.final_path)
        os.mkdir(self.final_path)
        if DEBUG:
            cv.namedWindow("img", cv.WINDOW_NORMAL)
            cv.resizeWindow("img", 1280, 960)
        for fp in data:
            img = cv.imread(fp, cv.IMREAD_GRAYSCALE)
            #th, bin_img= cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            img_cp = img.copy()
            self.Point_Noise_remove(img_cp, MIN_AREA, ITER) ##消除面积过小的联通区域
            #kernel = np.ones((3, 3), np.uint8)
            #img_cp = cv.dilate(img_cp, kernel, iterations=20)
            #img_cp = cv.erode(img_cp, kernel, iterations=20)
            if DEBUG:
                res = np.hstack((img, img_cp))
                cv.imshow("img", res)
                cv.waitKey(0)
            cv.imwrite(self.final_path + '/' + str(index).zfill(3) + '.png', img_cp)
            index += 1



    def Point_Noise_remove(self, img, minArea, ITER):
    #消除面积过小的联通区域
        kernel = np.ones((3, 3), np.uint8)
        dilate = cv.dilate(img, kernel, iterations=ITER)
        #canny1 = cv.Canny(dilate, 100, 200)
        _, labels, stats, centroids = cv.connectedComponentsWithStats(dilate)
        for istat in stats:
            if istat[4] < minArea:
                cv.rectangle(img, tuple(istat[0:2]), tuple(istat[0:2] + istat[2:4]), 0, thickness=-1)  # 26
        return img


if __name__ == '__main__':
    myPro = myResProcess()
    myPro.resProcess(MIN_AREA=15000, ITER=20)