'''
Author: xv rg16xw@163.com
Date: 2022-12-29 10:24:24
LastEditors: xv rg16xw@163.com
LastEditTime: 2022-12-29 20:22:22
FilePath: \opencv\filter.py
Description: filter2D
'''
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./11.png')
'''
filter2D滤波，先定义kernel卷积核参数
卷积核参数有几种经验核
'''

# 权重大于1
kernel_sharpen1 = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
# 权重等于1
kernel_sharpen2 = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])
# 权重小于1
kernel_sharpen3 = np.array([
        [-1, -1, -1],
        [-1, 7 -1],
        [-1, -1, -1] 
    ])
kernel4 = np.array([
        [-1, 0, -1],
        [0, 7 ,0],
        [1, 0, 1] 
    ])
kernel_edge_enhance = np.array([[-1,-1,-1,-1,-1],
                               [-1,2,2,2,-1],
                               [-1,2,8,2,-1],
                               [-2,2,2,2,-1],
                               [-1,-1,-1,-1,-1]])/8.0
'''
img为输入图像，
-1为第二个参数ddepth最深深度，-1说明要和原图深度一样，
kernel为是一部确定好的卷积核的大小和样子，
anchor锚点参数，默认根据核的形状找到中心位置作为锚点，
delta参数就是要不要在卷积之后加个偏移，默认加0，
borderType就是边缘类型，可以填充黑边，或者对得到的边缘做个什么反转等，默认BORDER_REFLECT_101：边缘反射101，对称法，以最边缘像素为轴，对称填充
'''

kernel=kernel_sharpen2

filter2D = cv2.filter2D(img, -1, kernel)

plt.subplot(121) 
plt.imshow(img)
plt.title('origin img')
plt.subplot(122) 
plt.imshow(filter2D)
plt.title('filter2D kernel is{}'.format(kernel))
plt.show()