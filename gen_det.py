import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import argparse

def binary(image):
    threshold_l=5
    arr1 = np.zeros(image.shape, np.uint8)
    th_x = image.shape[0]//4
    th_y = image.shape[1]//4
    mean = np.mean(image[th_x:3*th_x, th_y:3*th_y])
    std = np.std(image[th_x:3*th_x, th_y:3*th_y])
    temp = image[th_x:3*th_x, th_y:3*th_y].ravel()
    count = Counter(temp).most_common()
    threshold_r = max(62, 0.75*count[1][0], mean-0.78*std)
    if std > 50:
        threshold_r = mean-0.78*std
    if std < 35:
        threshold_r = 0.75*count[1][0]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i < th_x or i > 3*th_x or j < th_y or j > 3*th_y:
                continue
            if image[i][j] == 0:
                arr1[i][j] = 255
            elif image[i][j] > threshold_r or image[i][j] < threshold_l:
                continue
            else:
                arr1[i][j] = 255
    return arr1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generation Script for segmentation of Gall Bladder Images')
    parser.add_argument('-i', '--img_path', type=str, default='img', required=True, help="Path for the input image folder")
    parser.add_argument('-d', '--det_path', type=str, default='det', required=True, help="Path for the folder to save masks")
    
    args = parser.parse_args()
    for i in range(10):
        img1 = cv2.imread(args.img_path+"000{}.jpg".format(i), flags=cv2.IMREAD_GRAYSCALE)
        img1 = cv2.medianBlur(img1, 7)
        img2 = binary(img1).astype(np.uint8)
        img2 = cv2.bitwise_not(img2)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(img2), connectivity=8)
        sizes = list(stats[1:, -1])
        sizes = {sizes[i]: i for i in range(len(sizes))}
        sizes = list(dict(sorted(sizes.items(), key=lambda x: x[0], reverse=True)).values())
        nb_components -= 1
        for j in range(len(sizes)):
            temp = np.zeros((output.shape))
            temp[output == sizes[j]+1] = 255
            temp = temp.reshape(img2.shape).astype(np.uint8)
            if max(temp[:, (img2.shape[1]//2)]) == 255:
                drawing = np.zeros(img2.shape, np.int8)
                parts = 7
                half_col = img2.shape[1]//parts
                for h in range(parts):
                    im2, contours, hierarchy = cv2.findContours(temp[:, h*half_col:(h+1)*half_col], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) != 0:
                        for hu in range(len(contours)):
                            hull = [cv2.convexHull(contours[hu], False, clockwise=False)]
                            drawing[:, h*half_col:(h+1)*half_col] = cv2.fillConvexPoly(drawing[:, h*half_col:(h+1)*half_col], hull[0], (255, 255, 255))
                final = np.zeros(img2.shape, np.uint8)
                for row in range(drawing.shape[0]):
                    for col in range(drawing.shape[1]):
                        if row < img2.shape[0]//4 or row > 3*img2.shape[0]//4 or col < img2.shape[1]//4 or col > 3*img2.shape[1]//4 or drawing[row][col] < 50:
                            pass
                        else:
                            final[row][col] = 255
                cv2.imwrite(args.det_path+"000{}.jpg".format(i), final)
                break