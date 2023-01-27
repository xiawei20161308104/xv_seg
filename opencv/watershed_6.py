'''
Version: 6.0（交付版本）
Author: xiawei
Date: 2022-12-27 21:10:10
LastEditors: xiawei
LastEditTime: 2022-12-28 09:18:18
Description: 在实现1.png(watershed_5)基础上完善项目
找到最大相似边缘，排除边缘以内，保留边缘之外。交付版本。
依然存在的问题：需要保留的边缘小区域和边缘之外的稍大区域，不可兼得。
'''

import numpy as np
import cv2 as cv2
import canny as edge


img = cv2.imread('./6.png')
name = 6
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
medianBlur = cv2.medianBlur(gray_img, 3)
# 测试证明中值滤波中间噪点少很多
# GaussianBlur = cv2.GaussianBlur(gray_img, (11, 11),0, 0)
threshold = cv2.threshold(medianBlur, 202, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite(
    'C://Users/rg16x/Desktop/seg_result/threshold{}.png'.format(name), threshold)
cv2.namedWindow('threshold', cv2.WINDOW_NORMAL)
cv2.imshow('threshold', threshold)


# 二值化分析连通域
component_res = cv2.connectedComponentsWithStats(threshold, 4, cv2.CV_32S)
retval, labels, stats, centroids = component_res
print('retval', retval)
print('labels', labels)
print('stats', stats)
print('centroids', np.size(centroids))

# 定义输出矩阵
output = np.zeros(gray_img.shape, dtype="uint8")

# TODO：想办法得到一个内测相似边缘,并且得到边缘的这种方式要有鲁棒性，得确保得到的边缘合适
# 找到最大边缘
counter = edge.find_similar_counters(img)[0]
print(counter)

# 遍历连通域
for i in range(0, retval):
    '''
    参数1: 轮廓
    参数2: 点
    参数3: 设置为true时，返回实际距离值【点到轮廓的最短距离】
    若返回值为正，表示点在多边形内部，返回值为负，表示在多边形外部，
    返回值为0，表示在多边形上
    设置为false时，返回 - 1、0、1三个固定值。若返回值为+1，表示点在多边形内部，返回值为-1，表示在多边
    形外部，返回值为0，表示在多边形上
    '''
    # 获取左上角坐标
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    # 背景被标识为0
    if (x == 0 and y == 0):
        continue
    # 找到在最大边缘之外的左上角坐标
    distance = cv2.pointPolygonTest(counter, np.asfarray([x, y]), False)
    # area = stats[i, cv2.CC_STAT_AREA]
    # print('area', i, area)
    area = stats[i, cv2.CC_STAT_AREA]
    if distance == -1 and area > 800:
        print(i, '区域面积被选中')
        # (labels == i)有0和1
        # res = np.isin((labels == i), [0, 1]).all()
        # print(res)
        # print('labels', labels == i)

        componentMask = (labels == i).astype("uint8") * 255
        # output = cv2.bitwise_or(output, componentMask)
        output = output+componentMask
cv2.imwrite(
    'C://Users/rg16x/Desktop/seg_result/output{}.png'.format(name), output)

print(output)
cnt_array = np.where(output, 0, 255)
print(np.size(output))
print(np.sum(cnt_array))
print("1d geshu", np.size(output)-np.sum(cnt_array)/255)
cv2.waitKey(0)
