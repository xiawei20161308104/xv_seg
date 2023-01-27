import numpy as np
import cv2 as cv2


class a:
    def function1(self):

        self.function2()
        return 0

    def function2():
        return 0


def find_similar_counters(img):
    # img = cv2.imread('./5.png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 测试证明中值滤波和高斯效果不错，中值好于高斯，滤波核11时效果最好但是滤掉了些许边缘
    # 因为后续配置画一个内部相似轮廓，所以选择一个小的滤波核
    medianBlur = cv2.medianBlur(gray_img, 3)
    # GaussianBlur = cv2.GaussianBlur(gray_img, (11, 11),0, 0)
    # 阈值试出来的201/202，颜色一样嘛，不同图片阈值不一样怎么处理
    # 这个阈值刚好分出白色外圈
    threshold = cv2.threshold(medianBlur, 60, 255,
                              cv2.THRESH_BINARY_INV)[1]
    # threshold = cv2.threshold(medianBlur, 203, 255,
    #                           cv2.THRESH_BINARY_INV)[1]
    # 膨胀去除中间噪点，但边也被膨胀丢失了部分，最后选择先腐蚀扩大边缘和内部噪点不膨胀，保边
    # 但是这一步会失真！！并不是检测到的像素值的范围！是否保留？
    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.dilate(threshold, kernel, iterations=70)

    # erode=cv2.dilate(threshold,(9,13),iterations=4)

    edge = cv2.Canny(erode, 20, 100)
    contours, hier = cv2.findContours(
        edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print('np.size(contours)', np.size(contours))
    # print('contours', contours)
    areas = []
    for k in range(len(contours)):
        area_k = cv2.contourArea(contours[k])
        areas.append(area_k)
        # print(np.array(area_k))
    max_id = np.argmax(np.array(areas))
    # print("max_id", max_id)
    # print("areas", areas)
    # print('np.size(contours[max_id])', np.size(contours[max_id]))
    # 这个就是相似轮廓坐标
    # print('contours[max_id]', contours[max_id])
    draw_img = cv2.drawContours(img, contours, max_id, (0, 0, 255), 2)
    # image_left = cv2.resize(draw_img, (draw_img.shape[1], draw_img.shape[0]),fx=0.4, fy=0.24)
    # cv2.namedWindow('threshold', cv2.WINDOW_NORMAL)
    # cv2.imshow('threshold', threshold)
    # cv2.namedWindow('erode', cv2.WINDOW_NORMAL)
    # cv2.imshow('erode', erode)
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow('img', img)
    cv2.namedWindow('draw_img', cv2.WINDOW_NORMAL)
    cv2.imshow('draw_img', draw_img)

    # contours_matrix = contour
    return contours


if __name__ == '__main__':
    img = cv2.imread('./5.png')
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow('img', img)
    a = find_similar_counters(img)  # 分水岭找边界
    # b = a.reshape(2349, 2)
    print("main函数的返回值", a)
    # print("main函数的返回值2", type(b))
    # print("main函数的返回值4", b.ndim)
    # print("main函数的返回值3", b.shape)
    cv2.waitKey()
