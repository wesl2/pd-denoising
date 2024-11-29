import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

# # # #nhwc 直观
# a = np.arange(18).reshape((3, 3, 2))
# print(a)
# print(a[:,0:3:2,0:2:2])








def normalize(data):
    return data/255.

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
    #img是用numpy存储???  c w h
    endc = img.shape[0] # 图像的通道数 c
    endw = img.shape[1] # 图像的高度 h = endw
    endh = img.shape[2] # 图像的宽度 w = endh
    # 按照窗口大小和步长切分初始 patch，得到 (c, patch_w, patch_h) 的结果
    
    #切片操作：start:end:step
    #start：切片的起始索引。
    #end：切片的结束索引（注意，这是一个左闭右开区间，即不包含 end）。
    #step：切片的步长，即每隔多少个索引取一个值。1是全选
    #到 endw - win + 1 位置就不可以再切片了
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    # 计算小块的总数 TotalPatNum = 切分后的宽度数量 × 切分后的高度数量
    TotalPatNum = patch.shape[1] * patch.shape[2]
    # 初始化结果矩阵 Y，存储所有小块
    # 形状为 (c, win*win, TotalPatNum)，即将每个小块拉平成长度为win*win的向量
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
     # 双重循环：遍历窗口内的每个像素点 (i, j)
    for i in range(win):# 窗口的高度方向
        for j in range(win):# 窗口的宽度方向
            # 提取当前窗口对应位置的子图像块
            # 从第 i 行和第 j 列开始，每隔 stride 行/列取值
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            # 将当前子图像块拉平，并存储到结果矩阵 Y 中
            # Y 的第 k 行对应窗口内第 (i, j) 个像素的位置
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
        # 将结果 Y 从 (c, win*win, TotalPatNum) 重新调整为 (c, win, win, TotalPatNum)
    return Y.reshape([endc, win, win, TotalPatNum])


    
if __name__ == "__main__":
    path = "D:\\academic\\648580_3.jpg"
    Img = cv2.imread(path) # h w c
    h = Img.shape[0]
    w = Img.shape[1]
    print(Img.shape[0])
    print(Img.shape[1])
    print(Img.shape[2])
    Img = cv2.resize(Img, (int(h), int(w)), interpolation=cv2.INTER_CUBIC)
    Img = np.transpose(Img, (2,0,1))
    Img = np.float32(normalize(Img))
    t = Im2Patch(Img,100,20)

    patches = []
 
    print(t.shape[3])
    for i in range(t.shape[3]):
        patches.append(t[:,:,:,i])

    print(len(patches))
    for i in range(len(patches)):
        p = np.transpose(patches[i], (1, 2, 0)) 
        if(i % 500 == 0):
            print(p[:])
            cv2.imshow("p:", p)
            cv2.waitKey(0)

    for i in range(5):
        print(i)

    # cv2.imshow("tpic", tpic)
    # cv2.waitKey(0)
