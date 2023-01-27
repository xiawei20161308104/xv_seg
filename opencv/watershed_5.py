'''
Version: 1.0
Author: xiawei
Date: 2022-12-27 21:10:10
LastEditors: xiawei
LastEditTime: 2022-12-28 09:18:18
Description: 在实现1.png(watershed_4)基础上改进参数不适用等问题
这一版本尝试了两种方案，1.找到最大相似边缘作为分割边界 2.在药片内部画矩形作为分割边界。实现了第一种 第二种暂时报错
'''

# 读取,灰度,高斯,二值化
import numpy as np
import cv2 as cv2
import canny as edge
img = cv2.imread('./5.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
medianBlur = cv2.medianBlur(gray_img, 3)
# 测试证明中值滤波中间噪点少很多
# GaussianBlur = cv2.GaussianBlur(gray_img, (11, 11),0, 0)
threshold = cv2.threshold(medianBlur, 202, 255,
                          cv2.THRESH_BINARY_INV)[1]
# 膨胀去除中间噪点，但边也被膨胀丢失了部分，最后选择先腐蚀扩大边缘和内部噪点不膨胀，保边
# TODO问题就是失真
# erode=cv2.erode(threshold,(7,7),iterations=7)
# dilate = cv2.dilate(erode, (5, 5), iterations=1)
# cv2.namedWindow('erode', cv2.WINDOW_NORMAL)
# cv2.imshow('erode', erode)
'''
cv2.namedWindow('threshold', cv2.WINDOW_NORMAL)
cv2.imshow('threshold', threshold)
cv2.namedWindow('erode', cv2.WINDOW_NORMAL)
cv2.imshow('erode', erode)
'''
# 对腐蚀之后的 二值化分析连通域
totalLabels, label_ids, stats, centroid = cv2.connectedComponentsWithStats(threshold,
                                                                           4,
                                                                           cv2.CV_32S)
# 定义输出矩阵
output = np.zeros(gray_img.shape, dtype="uint8")
xs = []
ys = []
ws = []
hs = []
areas = []
print('totalLabels', totalLabels)
for i in range(2, totalLabels):

    # 提取当前标签的连通分量统计信息
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    (cX, cY) = centroid[i]
    xs.append(x)
    ys.append(y)
    # ws.append(w)
    # hs.append(h)
    areas.append(areas)
    print('x', x)
    print('y', y)
    print('area', area)
    '''
    参数1: 轮廓
    参数2: 点
    参数3: 设置为true时，返回实际距离值【点到轮廓的最短距离】
    若返回值为正，表示点在多边形内部，返回值为负，表示在多边形外部，
    返回值为0，表示在多边形上
    设置为false时，返回 - 1、0、1三个固定值。若返回值为+1，表示点在多边形内部，返回值为-1，表示在多边
    形外部，返回值为0，表示在多边形上
    '''
    # TODO：想办法得到一个内测相似边缘,并且得到边缘的这种方式要有鲁棒性，得确保得到的边缘合适
    distance = cv2.pointPolygonTest(
        edge.find_similar_counters, centroid[i], False)


# 循环之外找到连通域最大值
x_max = np.array(xs).max()
x_min = np.min(np.array(xs))
y_max = np.max(np.array(ys))
y_min = np.min(np.array(ys))
# areas_max = np.max(np.array(areas))
print('x_max', x_max)
print('x_min', x_min)
print('y_max', y_max)
print('y_min', y_min)
# print('areas_max',areas_max)
# 在腐蚀图上面画出矩形
# delta = 135
# delta_y = 100
# rectangle = cv2.rectangle(threshold, (x_min+delta, y_min+delta_y),
#                           (x_max-delta, y_max-delta_y), (0, 0, 255), -1)
# cv2.namedWindow('rectangle', cv2.WINDOW_NORMAL)
# cv2.imshow('rectangle', rectangle)

# 矩形收缩偏移量
for i in range(2, totalLabels):

    (cX, cY) = centroid[i]
    # 质心在设定矩形范围之内的排除掉
    x_i = x >= x_min+delta and x <= x_max-delta
    y_i = y >= y_min+delta_y and y <= y_max-delta_y
    # TODO 这样做可能会把白色边缘小区域过滤掉
    area_i = area < 100
    if all((x_i, y_i, area_i)):
        print('在矩形内 过滤掉')
    else:
        print('在矩形外 保留')
        componentMask = (label_ids == i).astype("uint8") * 255
        output = cv2.bitwise_or(output, componentMask)
        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.imshow('output', output)
'''
img22 = img.copy()
canny_img = cv2.Canny(output, 150, 300)

# 查找轮廓api，原始图像，查找轮廓方式RETR_EXTERNAL为之查找外围，List从里到外从右到左，CCOMP有层级关系没搞懂
# 从大到小从右到左,这里的contours为根据CHAIN_APPROX_SIMPLE设定的点。
# 返回轮廓个数和坐标值
area2 = []
contours, _ = cv2.findContours(
    canny_img, cv2.TERM_CRITERIA_MAX_ITER, cv2.CHAIN_APPROX_SIMPLE)
for k in range(len(contours)):
    area_k = cv2.contourArea(contours[k])
    if (area_k > 6000):
        print('come in 222')
        area2.append(k)
# max_idx = np.argmax(np.array(area2))
# 最大像素为212129.5
# print("area2222", cv2.contourArea(contours[max_idx]))
# 绘制轮廓，-1为绘制所有轮廓，颜色，线宽。
# print('masker', _)
# all_img2 = cv2.drawContours(img, contours, -1, (0, 0, 255), 5)
# cv2.namedWindow('all_img2', cv2.WINDOW_NORMAL)
# cv2.imshow('all_img2', all_img2)
for j in area2:
    img2 = cv2.drawContours(img, contours, j, (0, 255, 0), 5)
    output = cv2.bitwise_or(img, img2)
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.imshow('output', output)

# cv2.namedWindow('canny_img', cv2.WINDOW_NORMAL)
# cv2.imshow('canny_img', canny_img)
cv2.waitKey()


cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img22)
'''
cv2.waitKey(0)
