import cv2
import matplotlib.pyplot as plt
import numpy as np


def binary(image, threshold_l=5):
    arr1 = np.ndarray(image.shape)
    th_x = image.shape[0]//4
    th_y = image.shape[1]//4
    mean = np.mean(image[th_x:3*th_x, th_y:3*th_y])
    std = np.std(image[th_x:3*th_x, th_y:3*th_y])
    threshold_r = mean-0.85*std
    # threshold_r = 67
    # print(mean-0.85*std)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i < th_x or i > 3*th_x or j < th_y or j > 3*th_y:
                arr1[i][j] = 0
                continue
            if image[i][j] == 0:
                arr1[i][j] = 255
            elif image[i][j] > threshold_r or image[i][j] < threshold_l:
                arr1[i][j] = 0
            else:
                arr1[i][j] = 255
    return arr1


for i in range(10):
    img1 = cv2.imread("val/img/000{}.jpg".format(i), flags=cv2.IMREAD_GRAYSCALE)
    img1 = cv2.medianBlur(img1, 7)
    img2 = binary(img1).astype(np.uint8)
    img2 = cv2.bitwise_not(img2)
    # kernel = np.ones((5, 5), np.int8)
    # img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
    # img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
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
            im2, contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hull = [cv2.convexHull(contours[0], False)]
            drawing = np.zeros(img2.shape, np.int8)
            temp1 = cv2.fillConvexPoly(drawing, hull[0], (255, 255, 255))
            temp2 = np.zeros(img2.shape, np.uint8)
            for row in range(temp1.shape[0]):
                for col in range(temp1.shape[1]):
                    if row < img2.shape[0]//4 or row > 3*img2.shape[0]//4 or col < img2.shape[1]//4 or col > 3*img2.shape[1]//4 or temp1[row][col] < 50:
                        pass
                    else:
                        temp2[row][col] = 255
            cv2.imwrite("results/masked_000{}.jpg".format(i), temp2)
            break
