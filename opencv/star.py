import numpy as np
import cv2 as cv2

img = cv2.imread('./lantern.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 测试证明中值滤波和高斯效果不错，中值好于高斯，滤波核11时效果最好但是滤掉了些许边缘
# 因为后续配置画一个内部相似轮廓，所以选择一个小的滤波核
# GaussianBlur = cv2.GaussianBlur(gray_img, (11, 11),0, 0)
# 阈值试出来的201/202，颜色一样嘛，不同图片阈值不一样怎么处理
# 这个阈值刚好分出白色外圈
threshold = cv2.threshold(gray_img, 90, 255,
                          cv2.THRESH_BINARY)[1]
# threshold = cv2.threshold(medianBlur, 203, 255,
#                           cv2.THRESH_BINARY_INV)[1]
# 膨胀去除中间噪点，但边也被膨胀丢失了部分，最后选择先腐蚀扩大边缘和内部噪点不膨胀，保边
# 但是这一步会失真！！并不是检测到的像素值的范围！是否保留？
kernel = np.ones((3, 3), np.uint8)
erode = cv2.dilate(threshold, kernel, iterations=2)
edge = cv2.Canny(erode, 20, 100)
contours, hier = cv2.findContours(
    edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
draw_img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.namedWindow('erode', cv2.WINDOW_NORMAL)
cv2.imshow('erode', erode)
cv2.namedWindow('draw_img', cv2.WINDOW_NORMAL)
cv2.imshow('draw_img', draw_img)
cv2.waitKey()
