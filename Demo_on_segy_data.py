import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import *
from utils import *
from denoiser import *
from PIL import Image
import scipy.io as sio
import segyio
import matplotlib.pyplot as plt
import random
parser = argparse.ArgumentParser(description="PD-denoising")
#model parameter
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--delog", type=str, default="logsdc", help='path of log and model files')
parser.add_argument("--mode", type=str, default="MC", help='DnCNN-B (B) or MC-AWGN-RVIN (MC)')
#tested noise type
parser.add_argument("--color", type=int, default=0, help='[0]gray [1]color')
parser.add_argument("--real_n", type=int, default=1, help='real noise or synthesis noise [0]synthetic noises [1]real noisy image wo gnd [2]real noisy image with gnd')
parser.add_argument("--spat_n", type=int, default=0, help='whether to add spatial-variant signal-dependent noise, [0]no spatial [1]Gaussian-possion noise')
#pixel-shuffling parameter
parser.add_argument("--ps", type=int, default=2, help='pixel shuffle [0]no pixel-shuffle [1]adaptive pixel-ps [2]pre-set stride')
#FIXME:wbin default 512
parser.add_argument("--wbin", type=int, default=64, help='patch size while testing on full images')
parser.add_argument("--ps_scale", type=int, default=1, help='if ps==2, use this pixel shuffle stride')
#down-scaling parameter
parser.add_argument("--scale", type=float, default=1, help='resize the original images')
parser.add_argument("--rescale", type=int, default=1, help='resize it back to the origianl size after downsampling')
#testing data path and processing
parser.add_argument("--test_data", type=str, default='Set12', help='testing data path')
parser.add_argument("--test_data_gnd", type=str, default='Set12', help='testing data ground truth path if it exists')
parser.add_argument("--cond", type=int, default=1, help='Testing mode using noise map of: [0]Groundtruth [1]Estimated [2]External Input')
#TODO: 给下面两行多加了101010的默认值
parser.add_argument("--test_noise_level", nargs = "+", default=[30,70,20,0.5,17,29]  ,type=int,help='input noise level while generating noisy images')
parser.add_argument("--ext_test_noise_level", nargs = "+", type=int, default=[10,10,10,10,10,10],help='external noise level input used if cond==2')
#refining on the estimated noise map
parser.add_argument("--refine", type=int, default=0, help='[0]no refinement of estimation [1]refinement of the estimation')
parser.add_argument("--refine_opt", type=int, default=1, help='[0]get the most frequent [1]the maximum [2]Gaussian smooth [3]average value of 0 and 1 opt')
#FIXME: 把zeroout的defaul从0改成1
parser.add_argument("--zeroout", type=int, default=0, help='[0]no zeroing out [1]zeroing out some maps')
parser.add_argument("--keep_ind", nargs = "+", type=int, help='[0 1 2]Gaussian [3 4 5]Impulse')
#output options
parser.add_argument("--output_map", type=int, default=0, help='whether to output maps')
#K 可以交互式调整以平衡细节和背景，从而提供灵活的降噪性能。k=1 用于更多地关注平坦区域以获得非常平滑的结果，k=0 用于获得更多纹理细节（默认）
parser.add_argument("--k", type=float, default=0.1, help='merging factor between details and background')
parser.add_argument("--out_dir", type=str, default="D:\\academic\\PD-Denoising-pytorch\\results\\beijing\\", help='path of output files')

opt = parser.parse_args()
#the limitation range of each type of noise level: [0]Gaussian [1]Impulse
limit_set = [[0,75], [0, 80]]


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

def gen_patches(file_path, patch_size, stride_x, stride_y, scales):
    """
    对单炮数据进行数据切片，需要先将单炮数据的数据和道头剥离。
    Args:
        file_path:地震道数据的文件路径。
        patch_size:切片数据的大小，都是方形所以高宽一致。
        stride_x:在地震道数据x方向的滑动步长。
        stride_y:在地震道数据y方向的滑动步长。
        scales:输入为列表，对数据进行放缩。
    Returns:返回一系列的小数据块
    """
    ###python要注意tab 尤其是函数内的多重循环
    shot_data = np.load(file_path)  # 加载npy数据
    #TODO: trace 和 time 哪个在前
    time_sample, trace_number = shot_data.shape  # 获取数据大小
    patches = []   # 生成空列表用于添加小数据块
    for scale in scales:  # 遍历数据的缩放方式
        time_scaled, trace_scaled = int(time_sample * scale), int(trace_number * scale)  # 缩放后取整
        shot_scaled = cv2.resize(shot_data, (trace_scaled, time_scaled), interpolation=cv2.INTER_LINEAR) # 获得缩放后的数据，采用双线性插值
        # 数据归一化处理
        shot_scaled = shot_scaled / abs(shot_scaled).max()  # 将数据归一化到(-1,1)
        # 从放缩之后的shot_scaled中提取多个patch
        # 计算x方向滑动步长位置
        s1 = 1
        while (patch_size + (s1-1)*stride_x) <= trace_scaled:
            s1 = s1 + 1
        # python中索引默认0开始，而且左闭右开。patch_size + (s1-1)*stride_x就是切片滑动时候的实际位置加1
        # 这里的s1算出来大了1
        strides_x = []  # 用于存储x方向滑动步长位置
        x = np.arange(s1-1)  # 生成0~s1-2的序列数字
        x = x + 1  # 将序列变成1~s1-1
        for i in x:
            s_x = patch_size + (i-1)*stride_x  # 计算每一次的步长位置(实际位置加1)
            strides_x.append(s_x)  # 添加到列表
        # 计算y方向滑动步长位置
        s2 = 1
        while (patch_size + (s2-1)*stride_y) <= time_scaled:
            s2 = s2 + 1
        strides_y = []
        y = np.arange(s2-1)
        y = y + 1
        for i1 in y:
            s_y = patch_size + (i1-1)*stride_y
            strides_y.append(s_y)
        #  通过切片的索引位置在数据中提取小patch
        for index_x in strides_y:  # x方向索引是patch的列
            for index_y in strides_x:  # y方向索引是patch的行
                patch = shot_scaled[index_x-patch_size: index_x, index_y-patch_size: index_y]
                patches.append(patch)
    return patches

def data_generator(data_dir, patch_size, stride_x, stride_y, scales):
    """
    对整个目录下的npy文件进行，数据的切片。
    Args:
        data_dir: 文件夹路径
        patch_size:切片数据的大小，都是方形所以高宽一致。
        stride_x:在地震道数据x方向的滑动步长。
        stride_y:在地震道数据y方向的滑动步长。
        scales:输入为列表，对数据进行放缩。
    Returns:总的切片数据
    """
    file_list = glob.glob(os.path.join(data_dir, '*npy'))
    data = []
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i], patch_size, stride_x, stride_y, scales)
        for patch in patches:
            data.append(patch)
    print("获得切片数量：{}".format(len(data)))
    return data


def cut(seismic_block, patch_size, stride_x, stride_y):
    """
    :param seismic_block: 地震数据
    :param patch_size: 切片大小
    :param stride_x: 横向切片步长，大小等于patch_size
    :param stride_y: 竖向切片步长，大小等于patch_size
    :return: 按照规则填充后，获得的切片数据(以列表形式存储)，高方向切片数量，宽方向切片数量
    """
    [seismic_h, seismic_w] = seismic_block.shape  # 获取地震数据块的高(seismic_block_h)和宽(seismic_block_w)
    # 对数据进行填充，确保可以完整切片
    # 确定宽方向填充后大小
    n1 = 1
    while (patch_size + (n1 - 1) * stride_x) <= seismic_w:
        # 判断长为patch_size,步长为stride_x在长为seismic_w的时候能滑动多少步
        n1 = n1 + 1
    # 循环结束后计算的patch_size + (n1-1)*stride_x) > seismic_w，在滑动整数步长的时候可以完全覆盖数据
    arr_w = patch_size + (n1 - 1) * stride_x
    # 确定高方向填充后大小
    n2 = 1
    while (patch_size + (n2 - 1) * stride_y) <= seismic_h:
        n2 = n2 + 1
    arr_h = patch_size + (n2 - 1) * stride_y
    # # 对seismic_block数据块的右方和下方进行填充，填充内容为0
    fill_arr = np.zeros((arr_h, arr_w), dtype=np.float32)
    fill_arr[0:seismic_h, 0:seismic_w] = seismic_block
    # 对数据填充后，我们切分的数据是填充后的数据
    # 计算arr_w方向滑动步长位置
    # python中索引默认0开始，而且左闭右开。patch_size + (n-1)*stride_x就是切片滑动时候的实际位置加1
    # 这里的n算出来大了1
    path_w = []  # 用于存储x方向滑动步长位置
    x = np.arange(n1)  # 生成[0~n1-1]的序列数字
    x = x + 1  # 将序列变成[1~n1]
    for i in x:
        s_x = patch_size + (i - 1) * stride_x  # 计算每一次的步长位置(实际位置加1)
        path_w.append(s_x)  # 添加到列表
    number_w = len(path_w)
    path_h = []
    y = np.arange(n2)
    y = y + 1
    for k in y:
        s_y = patch_size + (k - 1) * stride_y
        path_h.append(s_y)
    number_h = len(path_h)
    #  通过切片的索引位置在数据中提取小patch
    cut_patches = []
    for index_x in path_h:  # path_h索引是patch的行
        for index_y in path_w:  # path_w索引是patch的列
            patch = fill_arr[index_x - patch_size: index_x, index_y - patch_size: index_y]
            cut_patches.append(patch)
    return cut_patches, number_h, number_w, arr_h, arr_w

def combine(patches, patch_size, number_h, number_w, block_h, block_w):
    """
    完整数据用get_patches切分后，将数据进行还原会原始数据块大小
    :param patches: get_patches切分后的结果，以列表形式传入
    :param patch_size: 数据切片patch的大小
    :param number_h: 高方向切出的patch数量
    :param number_w: 宽方向切出的patch数量
    :param block_h: 地震数据块的高
    :param block_w: 地震数据块的宽
    :return: 还原后的地震数据块
    """
    # 将列表patch1中的数据取出，转换成二维矩阵。按照列表元素顺序拼接。
    # patch_size = int(patch_size)
    temp = np.zeros((int(patch_size),1), dtype=np.float32)  # 临时拼接矩阵，后面要删除
    # 取出patch1中的每一个元素，在列方向(axis=1)拼接
    for i in range(len(patches)):
        temp = np.append(temp, patches[i], axis=1)
    # 删除temp后，此时temp1的维度是 patch_size * patch_size*number_h*number_w
    temp1 = np.delete(temp, 0, axis=1)  # 将temp删除

    # 将数据变成 (patch_size*number_h) * (patch_size*number_w)
    test = np.zeros((1, int(patch_size*number_w)), dtype=np.float32)  # 临时拼接矩阵，后面要删除
    # 让temp1每隔patch_size/2*number_w列就进行一个换行操作
    for j in range(0, int(patch_size*number_h*number_w), int(patch_size*number_w)):
        test = np.append(test, temp1[:, j:j + int(patch_size*number_w)], axis=0)
    test1 = np.delete(test, 0, axis=0)  # 将test删除
    block_data = test1[0:block_h, 0:block_w]
    return block_data

def img_normalize(data):
    return data/255.

def main():
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
    # Build model
    print('Loading model ...\n')


    seismic_noise = read_segy_data('D:\\academic\\dncnn_pytorch\\data\\sgy_data\\synthetic.sgy')  # 野外地震数据        
    seismic_block_h, seismic_block_w = seismic_noise.shape
    # 数据归一化处理
    seismic_noise_max = abs(seismic_noise).max()  # 获取数据最大幅值
    seismic_noise = seismic_noise / seismic_noise_max  # 将数据归一化到(-1,1)  
    # 对缺失的炮集数据进行膨胀填充，并且切分
    patch_size = 64
    patches, strides_x, strides_y, fill_arr_h, fill_arr_w = cut(seismic_noise, patch_size, patch_size, patch_size)



    c = 1 
    net = DnCNN_c(channels=c, num_of_layers=opt.num_of_layers, num_of_est = 2 * c)
    est_net = Estimation_direct(c, 2 * c)

    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    #TODO:相对先改绝对 后面要看怎么改
    #model.load_state_dict(torch.load(os.path.join(opt.delog, 'net.pth')))
    model.load_state_dict(torch.load(os.path.join("D:\\academic\\PD-Denoising-pytorch\\logs\\logs_gray_MC_AWGN_RVIN", 'net.pth')))
    model.eval()

    #Estimator Model
    model_est = nn.DataParallel(est_net, device_ids=device_ids).cuda()
    #TODO:相对先改绝对 后面要看怎么改
    #model_est.load_state_dict(torch.load(os.path.join(opt.delog, 'est_net.pth')))
    model_est.load_state_dict(torch.load(os.path.join("D:\\academic\\PD-Denoising-pytorch\\logs\\logs_gray_MC_AWGN_RVIN", 'est_net.pth')))
    model_est.eval()

    # 检测是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict_datas = []  # 空列表，用于存放网络预测的切片数据
    # 对切片数据进行网络预测
    for patch in patches:
        patch = np.array(patch)  # 转换为numpy数据
        #patch = patch.reshape(1,patch.shape[0])  # 对数据维度进行扩充(批量，通道，高，宽)
        
        Img = patch  #input image with w*h*c
        Img = np.expand_dims(Img, axis=-1)
        #print(Img.shape)
        #patch = torch.from_numpy(patch)  # python转换为tensor
        #patch = patch.to(device=device, dtype=torch.float32)  # 数据拷贝至GPU
        #print(patch.shape)
        # image

        w, h , _= Img.shape
        Img = Img[:,:,::-1]  #change it to RGB
        print(Img.shape,"Img.shape")
        #Img = cv2.resize(Img, (0,0), fx=opt.scale, fy=opt.scale)
        # if opt.color == 0:
        #     Img = Img[:,:,0]  #For gray images
        #     Img = np.expand_dims(Img, 2)
        pss=1
        if opt.ps == 1:
            #TODO:相对先改绝对 后面要看怎么改
            #pss = decide_scale_factor(Img/255., model_est, color=opt.color,  thre = 0.008, plot_flag = 1, stopping = 4,mark = opt.out_dir + '/' +  file_name)[0]
            pss = decide_scale_factor(Img/255., model_est, color=opt.color,  thre = 0.008, plot_flag = 1, stopping = 4,mark = "D:\\academic\\PD-Denoising-pytorch\\results\\beijing" + '/' +  file_name)[0]
            print(pss)
            Img = pixelshuffle(Img, pss)
        elif opt.ps == 2:
            pss = opt.ps_scale
        
        merge_out= np.zeros([w,h,1])
        wbin = opt.wbin
        i = 0
        while i < w:
            i_end = min(i+wbin, w)
            j = 0
            while j < h:
                j_end = min(j+wbin, h)
                patch = Img[i:i_end,j:j_end,:]
                predict_data = denoiser(Img, c, pss, model, model_est, opt)
                merge_out[i:i_end, j:j_end, :] = predict_data        
                j = j_end
            i = i_end
        out = merge_out  # 将数据从GPU中拷贝出来，放入CPU中，并转换为numpy数组
        print(out.shape)
        predict_data = out.squeeze()  # 默认压缩所有为1的维度
        print(out.shape,"去掉维度为1之后")
        predict_datas.append(out)  # 添加至列表中

    predict_datas =np.squeeze(predict_datas)
    # 对预测后的数据进行还原，裁剪
    seismic_predict = combine(predict_datas, patch_size, strides_x, strides_y, seismic_block_h, seismic_block_w)
    # 数据逆归一化处理
    seismic_predict = seismic_predict*seismic_noise_max  # 将数据归一化到(-1,1)
    #  显示处理效果
    fig1 = plt.figure()
    # 三个参数分别为：行数，列数，
    ax1 = fig1.add_subplot(1, 3, 1)
    ax2 = fig1.add_subplot(1, 3, 2)
    ax3 = fig1.add_subplot(1, 3, 3)
    # 绘制曲线
    ax1.imshow(seismic_noise, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
    ax2.imshow(seismic_predict, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
    ax3.imshow(seismic_noise-seismic_predict, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
    plt.tight_layout()  # 自动调整子图位置
    plt.show()
    #cv2.imwrite(os.path.join("D:\\academic\\PD-Denoising-pytorch\\results_bc\\", file_name + '_pss'+str(pss)+'_k'+str(opt.k)+'.png'), merge_out[:,:,::-1])
    #path_t = os.path.join(opt.out_dir, file_name + '_pss'+str(pss)+'_k'+str(opt.k)+'.png')
    path_t = "D:\\academic\\PD-Denoising-pytorch\\results\\beijing"
    iname = "i = {}".format(t) + '_pss'+str(pss)+'_k'+str(opt.k)+'.png'
    print(file_name)
    print(path_t)
    path_t = os.path.join(path_t,iname)
    #cv2.imwrite(path_t, merge_out[:,:,::-1])
    cv2.imwrite(path_t,out_image)
    print('done!')



if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
