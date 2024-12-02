import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch
import segyio
import matplotlib.pyplot as plt
import random
from utils import *
import argparse
# # nl_list = np.zeros((1,3,1))
# # print(nl_list)


# # #nhwc 直观
# # a = np.ones((2,1,10))

# # print(a[2,1,:])
# # print(a.shape[0])

# r = np.random.uniform(1,20,size=10).astype(int)
# print(r)
# w,h = np.histogram(r,density=True,bins=20)
# print(w,h)
# img = plt.bar(h[:-1], w) # bin_edges的长度是hist长度 + 1 故舍弃bin_edges数组最后一个数值
# plt.show()

# w,h = np.histogram(r,normed=True,bins=10)
# plt.bar(h[:-1], w) # bin_edges的长度是hist长度 + 1 故舍弃bin_edges数组最后一个数值
# plt.show()
# nl_list = np.zeros((1,3,1))
# print(nl_list)


# #nhwc 直观
# a = np.ones((2,1,10))

# print(a[2,1,:])
# print(a.shape[0])

def pixelshuffle(image, scale):
    '''
    Discription: Given an image, return a reversible sub-sampling
    [Input]: Image ndarray float
    [Return]: A mosic image of shuffled pixels
    描述：给定一张图像，返回可逆的子采样图像
    [输入]：图像 ndarray（浮点型）
    [返回]：一个像素打乱后的马赛克图像
    '''
    # 如果缩放因子为 1，直接返回原图像
    if scale == 1:
        return image
    w, h ,c = image.shape# 获取图像的宽度、高度和通道数
    mosaic = np.array([])  # 初始化马赛克图像数组
    # 对于每个 scale 的宽度进行迭代
    for ws in range(scale):
        band = np.array([]) # 初始化带（行）的数组
        # 对于每个 scale 的高度进行迭代
        for hs in range(scale):
            # 获取经过子采样的图像，ws 和 hs 是行和列的起始索引
            temp = image[ws::scale, hs::scale, :]  #get the sub-sampled image
             # 如果 band 数组为空，则直接赋值；否则将新提取的图像拼接到 band 中
            band = np.concatenate((band, temp), axis = 1) if band.size else temp
        # 将 band 拼接到马赛克图像 mosaic 中
        mosaic = np.concatenate((mosaic, band), axis = 0) if mosaic.size else band
    return mosaic

def Im2Patch(img, win, stride=1):
    '''
    Description: 将输入图像切分为小块（patches），每个小块大小为 win × win，切分的步长为 stride
    ----------
    [输入]
    img: 输入图像，形状为 (c, h, w)，其中 c 是通道数，h 和 w 是图像高度和宽度
    win: 切分小块的窗口大小
    stride: 切分小块的步长（默认为 1）

    [输出]
    Y: 切分后的图像小块，形状为 (c, win, win, TotalPatNum)，其中 TotalPatNum 是小块的总数
    '''
    k = 0
    endc = img.shape[0] # 图像的通道数 c
    endw = img.shape[1] # 图像的宽度 w
    endh = img.shape[2] # 图像的高度 h
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def read_segy_data(filename):
    """
    读取segy或者sgy数据，剥离道头信息
    :param filename: segy或者sgy文件的路径
    :return: 不含道头信息的地震道数据
    """
    print("### Reading SEGY-formatted Seismic Data:")
    print("Data file-->[%s]" %(filename))
    with segyio.open(filename, "r", ignore_geometry=True)as f:
        f.mmap()
        data = np.asarray([np.copy(x) for x in f.trace[:]]).T
    f.close()
    return data

if __name__ == "__main__":
    # path = "D:\\academic\\648580_3.jpg"
    # Img = cv2.imread(path)
    # tpic = pixelshuffle(Img,5)
    # cv2.imshow("tpic", tpic)
    # cv2.waitKey(0)

    # seismic_noise = read_segy_data('D:\\academic\\dncnn_pytorch\\data\\sgy_data\\synthetic.sgy')  # 野外地震数据        
    # seismic_noise = seismic_noise[:, :,np.newaxis] 
    # seismic_noise = np2ts(seismic_noise)
    # print(seismic_noise.shape) 
    # out_numpy = visual_va2np(seismic_noise, 0, 0, 1, 1, 1, 1501, 9030, 1)


    path_t = "D:\\academic\\PD-Denoising-pytorch\\results\\beijing"
    iname = "OUT"+'.png'

    # print(path_t)
    # path_t = os.path.join(path_t,iname)
    # #cv2.imwrite(path_t, merge_out[:,:,::-1])
    # cv2.imwrite(path_t,out_numpy)

    img1 = cv2.imread("D:\\academic\\PD-Denoising-pytorch\\results\\beijing\\atest_show.png")
    img2 = cv2.imread("D:\\academic\\PD-Denoising-pytorch\\results\\beijing\\result.png")

    iout = img1 - img2
    print(iout.shape)
    path_t = os.path.join(path_t,iname)
    cv2.imwrite(path_t, iout)


    #fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制噪声数据
    # axes[0].imshow(seismic_noise, cmap=plt.cm.seismic, aspect='auto', vmin=vmin, vmax=vmax)
    # axes[0].set_title('Noise Data')
    # axes[0].axis('off')  # 关闭坐标轴
    # plt.tight_layout()  # 自动调整子图位置
    # plt.show()
    # 数据归一化处理
    #seismic_noise_max = abs(seismic_noise).max()  # 获取数据最大幅值
    #seismic_noise = seismic_noise / seismic_noise_max  # 将数据归一化到(-1,1) 
    #print(seismic_noise.shape)
    #print(seismic_noise)

    #seismic_noise = seismic_noise[:, :,np.newaxis] 
    #print(seismic_noise.shape)


# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# temp = a[0::2, 0::2]
# print(temp)
# temp = a[0:2,0:2]
# print(temp)
# print(a[1])

# a = torch.tensor([[2,1],[2,1]])
# print(a)
# a = torch.unsqueeze(a,dim=2)
# print(a)
