import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import math
import argparse

class Block_Matching():
    def __init__(self, img1data, img2data, window_size, start, search_range, output):
        h, w = img1data.shape[:2]
        self.width = w # width size of image
        self.height = h # height size of image
        self.winsize = window_size # window size should become odd number
        self.img1 = img1data # data of left image
        self.img2 = img2data # data of right image
        self.start = start
        self.search_range = search_range
        self.output = output

    def MSE(self, img1data, img2data):
        difference = np.subtract(img1data, img2data, dtype=np.int16)
        loss = np.sum(difference**2)
        return loss / (len(img1data)**2)

    def SAD(self, img1data, img2data):
        difference = np.subtract(img1data, img2data, dtype=np.int16)
        loss = 0
        for i in range(len(img1data)):
            loss += abs(difference[i])
        return loss

    def NCC(self, img1data, img2data):
        mean1 = np.mean(img1data, dtype=np.int16)
        mean2 = np.mean(img2data, dtype=np.int16)

        i1feature = (img1data-mean1)/math.sqrt(np.sum(img1data-mean1)**2)
        i2feature = (img2data-mean2)/math.sqrt(np.sum(img2data-mean2)**2)
        return np.sum(i1feature*i2feature)

    def SSD(self, img1data, img2data):
        return np.sum((img1data-img2data)**2, dtype=np.int16)

    def CENSUS(self, img1data, img2data):

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
                idx += 1
            else :
                idx += 1

        compare = census1 ^ census2
        return np.count_nonzero(compare)

    def Matching(self):
        step = 0
        print("height : {}, width : {}".format(self.height, self.width))
        calculation_result = np.zeros((self.search_range), dtype=np.float)
        disparity_map = np.zeros((self.height, self.width), dtype=np.float)
        for y in range(self.height - self.winsize -1):
            print('step : {}'.format(step))
            for x in range(self.width - self.winsize - 50):
                img1feature_point = self.img1[y : y + self.winsize, x : x + self.winsize]
                img1feature_point = np.reshape(img1feature_point, (self.winsize ** 2))

                #if self.width - self.winsize - x > 57 :
                for d in range(self.width - self.winsize - x -50):
                    #print('y : {}, x : {}, d : {}, d range : {}'.format(y, x, d, self.width - self.winsize - x - 50))
                    img2feature_point = self.img2[y : y + self.winsize, x + self.start + d : x + self.start + d + self.winsize]
                    img2feature_point = np.reshape(img2feature_point, (self.winsize ** 2))

                    loss_value = self.MSE(img1feature_point, img2feature_point)
                    calculation_result[d]=loss_value
                    # print('{}'.format(np.isnan(np.min(calculation_result))))

                    if d == self.search_range-1:
                        break

                disparity = np.argmin(calculation_result) + 50
                disparity_map[y+int(self.winsize/2), x+int(self.winsize/2)] = disparity
            step += 1

        np.save(self.output, disparity_map)
        plt.imshow(disparity_map, 'gray')
        plt.show()


    def ErrorofDisparity(self, data, y_value):
        plt.plot(data[1:-1])
        plt.xlabel('Disparity')
        plt.ylabel('Error')
        fname = './asset/error_of_disparity_graph_case_by_case/'+str(y_value)+'line_disparity_image'
        plt.savefig(fname+'.png')
        plt.clf()


def main():
    parser = argparse.ArgumentParser(description = "Read left and right image with stereo pair and calculate the disparity map")
    parser.add_argument("--IMG_DIR", dest='IMG_DIR', help="path of image directory")
    parser.add_argument("--Compute_Opt", dest='Compute_Opt', help="compute option, calculate disparity map : 'DISP' <or> calculate depth map : 'DEPTH' ")
    parser.add_argument('--WIN_SIZE', dest='WIN_SIZE', help="size of window for calculate the disparity")
    parser.add_argument("--Start", dest='Start', help="start point")
    parser.add_argument("--Search_Range", dest='Search_Range', help="range of searching disparity")
    parser.add_argument("--focal_Length", dest='focaL_Length', help="focal length")
    parser.add_argument("--Base_Length", dest='Base_Length', help="length of base line")
    parser.add_argument("--Outputfile", dest="output", help="path of output file")
    args = parser.parse_args()

    print(args.Compute_Opt)
    if args.Compute_Opt == 'DISP':
        img_list_for_matching = os.listdir(args.IMG_DIR)
        img_data = []

        for i in img_list_for_matching:
            img_data.append(cv2.imread(args.IMG_DIR+'/'+i, 0))

        block_matcher = Block_Matching(img1data=img_data[0],
                                       img2data=img_data[1],
                                       window_size=int(args.WIN_SIZE),
                                       start=int(args.Start),
                                       search_range=int(args.Search_Range),
                                       output=args.output)

        block_matcher.Matching()

    elif args.Compute_Opt == "DEPTH":
        pass

if __name__ =="__main__":
    # disparity_map = np.load("SAD7.npy")
    # plt.imshow(disparity_map, 'gist_heat')
    # plt.show()
    main()


