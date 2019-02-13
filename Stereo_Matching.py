import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import math

M_IMGDIR = './asset/matching/'
C_IMGDIR = './asset/color/'

img_list_for_matching = os.listdir(M_IMGDIR)
img_data = []

for i in img_list_for_matching:
    img_data.append(cv2.imread(M_IMGDIR+i, 0))


class Block_Matching():
    def __init__(self, img1data, img2data, window_size):
        w, h = img1data.shape[:2]
        self.width = w # width size of image
        self.height = h # height size of image
        self.winsize = window_size # window size should become odd number
        self.img1 = img1data # data of left image
        self.img2 = img2data # data of right image

    def MSE(self, img1data, img2data):
        #print("data in the block of img1 : {}".format(img1data))
        #print("data in the block of img2 : {}".format(img2data))
        difference = np.subtract(img1data, img2data, dtype=np.int16)
        #print("difference between two blocks : {}".format(difference))
        loss = 0
        for i in range(len(img1data)):
            loss += ((difference[i])**2)
        return loss / (len(img1data)**2)

    def SAD(self, img1data, img2data):
        print("data in the block of img1 : {}".format(img1data))
        print("data in the block of img2 : {}".format(img2data))
        difference = np.subtract(img1data, img2data, dtype=np.int16)
        print("difference between two blocks : {}".format(difference))
        loss = 0
        for i in range(len(img1data)):
            loss += abs(difference[i])
        return loss

    def NCC(self, img1data, img2data):
        print("data in the block of img1 : {}".format(img1data))
        print("data in the block of img2 : {}".format(img2data))
        mean1 = np.mean(img1data, dtype=np.int16)
        mean2 = np.mean(img2data, dtype=np.int16)

        denominator = 0
        numerator = 0
        for i in range(len(img1data)):
            denominator += ((img1data[i]-mean1)*(img2data[i]-mean2))
            numerator += (((img1data[i]-mean1)**2)*((img2data[i]-mean2)**2))
        return denominator/math.sqrt(numerator)

    def SSD(self, img1data, img2data):
        print("data in the block of img1 : {}".format(img1data))
        print("data in the block of img2 : {}".format(img2data))
        return np.sum((img1data-img2data)**2, dtype=np.int16)

    def CENSUS(self, img1data, img2data):
        print("data in the block of img1 : {}".format(img1data))
        print("data in the block of img2 : {}".format(img2data))

        mid_idx = int((self.winsize **2)/2)

        census1 = np.zeros((len(img1data)), dtype='uint8')
        census2 = np.zeros((len(img2data)), dtype='uint8')

        idx = 0
        for i in range(len(img1data)):
            if i == mid_idx:
                continue
            elif img1data[i] > img1data[mid_idx]:
                census1[idx] = 1
                idx += 1
            else :
                idx += 1
        idx = 0
        for i in range(len(img2data)):
            if i == mid_idx:
                continue
            elif img2data[i] > img2data[mid_idx]:
                census2[idx] = 1
            else :
                idx += 1

        compare = census1 ^ census2
        return np.count_nonzero(compare)

    def Matching(self):
        calculation_result = np.zeros((self.width, self.height), dtype=np.float)
        print("size of point matrix : {}".format(np.shape(calculation_result)))
        for y in range(self.height - self.winsize):
            #print('check original : {}' .format(self.img1[y][0:3]))
            #print('check original another : {}'.format(self.img1[y:y+3,0:3]))
            for x in range(self.width - self.winsize):
                for d in range(self.width - self.winsize - x):
                    img1feature_point = self.img1[y:y+self.winsize,x:x+self.winsize]
                    #print('form of feature of img1 : {}'.format(img1feature_point))
                    img1feature_point = np.reshape(img1feature_point, (self.winsize**2))
                    img2feature_point = self.img2[y:y+self.winsize,x+d:x+d+self.winsize]
                    #print('form of feature of img2 : {}'.format(img2feature_point))
                    img2feature_point = np.reshape(img2feature_point, (self.winsize**2))
                    loss_value = self.MSE(img1feature_point, img2feature_point)
                    #print('calculated loss value : {}'.format(loss_value))
                    calculation_result[y+int(self.winsize/2),x+d+int(self.winsize/2)]=loss_value
                    #if d == 10:
                    #    exit()
            if y <= self.height - self.winsize-1:
                self.ErrorofDisparity(calculation_result[y+int(self.winsize/2)],y+int(self.winsize/2))
                #print(calculation_result[y+int(self.winsize/2)])

    def ErrorofDisparity(self, data, y_value):
        plt.plot(data[1:-1])
        plt.xlabel('Disparity')
        plt.ylabel('Error')
        fname = './asset/error_of_disparity_graph_case_by_case/'+str(y_value)+'line_disparity_image'
        plt.savefig(fname+'.png')
        plt.clf()

block_matcher = Block_Matching(img1data=img_data[0], img2data=img_data[1], window_size=3)
block_matcher.Matching()