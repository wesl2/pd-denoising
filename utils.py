import math
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
#from skimage.measure.simple_metrics import compare_psnr
from torch.autograd import Variable
import cv2
import scipy.ndimage
import scipy.io as sio
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def weights_init_kaiming(m):
    classname = m.__class__.__name__ # 获取该层的类名，用于判断层的类型
    
    # 递归地遍历每个网络层，并根据层的类型进行不同的权重初始化
    #apply: 递归到每一个模块 进行权重初始化
    # 该层是卷积层：正向传播方差为 0 的kaiming初始化
    # 该层是全连接层：正向传播方差为 0 的kaiming初始化
    # 该层是标准归一化层：
    # a为激活函数的负半轴的斜率，若为relu则是 0


    # 如果该层是卷积层 (类名中包含 'Conv')
    # 使用 Kaiming 正态分布初始化，适用于 ReLU 激活函数
    # a=0 表示 ReLU，没有负泄露参数，负半轴的斜率为0，fan_in 模式表示只考虑输入的方差
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    # 如果该层是全连接层 (类名中包含 'Linear')
    # 同样使用 Kaiming 正态分布初始化，适用于 ReLU 激活函数
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    # 如果该层是批归一化层 (类名中包含 'BatchNorm')
    # 对权重执行正态分布初始化，并将权重值限制在 [-0.025, 0.025] 之间
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        # 对偏置 (bias) 执行常数初始化，全部初始化为 0
        nn.init.constant(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    # 将 img (估计的图像) 和 imclean (干净的图像) 从 GPU 转移到 CPU，并转换为 NumPy 数组
    # .data: 获取张量的数据部分
    # .cpu(): 如果张量在 GPU 上，移动到 CPU
    # .numpy(): 将 PyTorch 张量转换为 NumPy 数组，方便后续处理
    # .astype(np.float32): 将数据类型转换为 32 位浮点数，确保计算精度
    Img = img.data.cpu().numpy().astype(np.float32)
    
    ##Img = img.data.to(device=CPU,dtype=float32)
    
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    # 初始化 PSNR 值为 0，用于累计计算每个图像的 PSNR
    # 遍历每张图片，逐个计算 PSNR
    # Img.shape[0] 表示批量的大小（即有多少批图像）
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])
#PSNR 解释：
#PSNR（峰值信噪比）：用于衡量图像的质量，尤其在图像去噪、压缩等任务中，
# 用来评估重建图像与参考图像之间的差异。
# PSNR 越高，代表重建图像与参考图像的质量越接近。


def data_augmentation(image, mode):
    """
    data augmentation 数据扩充
    :param img: 二维矩阵
    :param mode: 对矩阵的翻转方式
    :return: 翻转后的矩阵
    """
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))


#TODO: notice
#用opencv读取图片 需要转置其格式从HWC到CHW
def visual_va2np(Out, mode=1, ps=0, pss=1, scal=1, rescale=0, w=10, h=10, c=3, refill=0, refill_img=0, refill_ind=[0, 0]):
     # 根据 mode 的值处理输出张量，转换为 NumPy 数组
    if mode == 0 or mode == 1 or mode==3:
        out_numpy = Out.data.squeeze(0).cpu().numpy()# 将 PyTorch 张量 squeeze（压缩掉第一个维度），移到 CPU 上并转换为 NumPy 数组
    elif mode == 2:
        out_numpy = Out.data.squeeze(1).cpu().numpy() # 在 mode == 2 时，压缩掉第二个维度
    if out_numpy.shape[0] == 1:
        out_numpy = np.tile(out_numpy, (3, 1, 1))# 如果 NumPy 数组的第一个维度是 1（灰度图），将其扩展为 3 个通道（RGB）
    # 根据 mode 的值进行不同的处理
    if mode == 0 or mode == 1:
        # 转置 NumPy 数组的维度（从 (C, H, W) 到 (H, W, C)），方便进行图像操作
        # 并将像素值放大到 255 范围（0-255），通过 scal 缩放系数调整
        out_numpy = (np.transpose(out_numpy, (1, 2, 0))) * 255.0 * scal
    else:
        out_numpy = (np.transpose(out_numpy, (1, 2, 0))) # 如果 mode 不为 0 或 1，则只进行维度转置，不进行放大
    # 如果 ps == 1，则进行 pixel shuffle 反变换（图像复原操作）    
    if ps == 1:
        # 使用 reverse_pixelshuffle 函数对图像进行反 pixel shuffle 处理
        out_numpy = reverse_pixelshuffle(out_numpy, pss, refill, refill_img, refill_ind)
    if rescale == 1:
        #如果 rescale == 1，则对图像进行尺寸调整，使用 OpenCV 的 resize 函数将图像缩放到 (h, w)
        out_numpy = cv2.resize(out_numpy, (h, w))
        #可选调试输出图像大小（这行注释掉的代码可能用于调试）
        #print(out_numpy.shape)
    return out_numpy

def temp_ps_4comb(Out, In):
    pass

#TODO:NCHW与NHWC
# region CHW and HWC

#本身图像读取后转成numpy格式是HWC的，但是pytorch大部分处理函数都是CHW，所以需要转换
#HWC格式
#HWC格式是指按照高度、宽度和通道数的顺序排列图像尺寸的格式。
# 例如，一张形状为256×256×3的RGB图像，在HWC格式中表示为[256, 256, 3]。
# 在一些图像处理库或者底层框架中，例如OpenCV和TensorFlow，通常使用HWC格式表示图像尺寸。
#在OpenCV中，读取的图片默认是HWC格式，即按照高度、宽度和通道数的顺序排列图像尺寸的格式。
# 例如，一张形状为256×256×3的RGB图像，在OpenCV中读取后的格式为[256, 256, 3]，其中最后一个维度表示图像的通道数。
# 在OpenCV中，可以通过cv2.imread()函数读取图片，该函数的返回值是一个NumPy数组，表示读取的图像像素值。
# 需要注意的是，OpenCV读取的图像像素值是按照BGR顺序排列的，而不是RGB顺序。
# 因此，如果需要将OpenCV读取的图像转换为RGB顺序，可以使用cv2.cvtColor()函数进行转换。

# CHW格式
# CHW格式是指按照通道数、高度和宽度的顺序排列图像尺寸的格式。
# 例如，一张形状为3×256×256的RGB图像，在CHW格式中表示为[3, 256, 256]。
# 在计算机视觉和深度学习中，通常使用CHW格式表示图像尺寸。
# 在PyTorch中，模型接收的RGB图像通常采用CHW格式，即按照通道数、高度和宽度的顺序排列像素信息的方式。
# 在CHW格式中，每个像素点的RGB值依次排列在内存中，通道数是第一维，高度是第二维，宽度是第三维。
# 因此，对于一个形状为[C, H, W]的RGB图像，C表示通道数，通常为3，H表示高度，W表示宽度。
# 对于每个像素点，其RGB值依次存储在内存中相邻的位置上。

# HWC格式和CHW格式虽然表示方式不同，但它们可以互相转换。
# 对于一张形状为[H, W, C]的图像，我们可以使用transpose函数将其转换为形状为[C, H, W]的图像，即CHW格式。转换方法如下：

# import numpy as np
# # 创建一个形状为[256, 256, 3]的随机图像
# img = np.random.rand(256, 256, 3)
# # 将HWC格式的图像转换为CHW格式
# img_chw = np.transpose(img, (2, 0, 1))
# 运行运行
# 另外，对于一些深度学习框架，如PyTorch和Caffe2等，通常要求输入的图像张量格式为CHW格式。
# 因此，在使用这些框架进行图像处理时，需要将图像张量从HWC格式转换为CHW格式。
# endregion

def np2ts(x, mode=0):  #now assume the input only has one channel which is ignored
    # 当前假设输入只有一个通道（忽略通道数）
    # 获取输入 NumPy 数组的形状，w 表示宽度，h 表示高度，c 表示通道数
    # mode 代表 color  color=1（mode=1） 有3通道    color = 0（mode=0） 有1通道
    w, h, c= x.shape
    x_ts = x.transpose(2, 0, 1)
    x_ts = torch.from_numpy(x_ts).type(torch.FloatTensor) 
    # 将 NumPy 数组转换为 PyTorch 张量，并将数据类型转换为 FloatTensor（浮点数张量）

    # 根据 mode 参数决定是否添加额外的维度
    # mode == 0 或 mode == 1 时（RGB或者灰度图时），将在第 0 维度前增加一个维度，通常表示批量维度
    if mode == 0 or mode == 1:
        x_ts = x_ts.unsqueeze(0) #在第 0 维增加一个维度，变为 (1, C, H, W)
    elif mode == 2:    # mode == 2 时，将在第 1 维度后增加一个维度，通常表示输入通道
        x_ts = x_ts.unsqueeze(1) # 在第 1 维增加一个维度，变为 (C, 1, H, W) 适合某些特定的网络输入需求。
    return x_ts

def np2ts_4d(x):
    # 将输入的 NumPy 数组的维度从 (N, H, W, C) 转换为 (N, C, H, W)
    # 其中 N 表示批量大小 即这批图像有几张，H 是高度，W 是宽度，C 是通道数
    x_ts = x.transpose(0, 3, 1, 2)
    x_ts = torch.from_numpy(x_ts).type(torch.FloatTensor)
    return x_ts

def get_salient_noise_in_maps(lm, thre = 0., chn=3):
    '''
    salient:突出的
    chn = channel
    Description: To find out the most frequent estimated noise level in the images
    描述：找出图像中最频繁的估计噪声水平 即 第n张图第c个通道的噪声最频繁值
    ----------
    [Input]
    a multi-channel tensor of noise map
    一个多通道的噪声图张量
    [Output]
    A list of noise level value
    一个噪声水平值的列表
    '''
    lm_numpy = lm.data.cpu().numpy()# 将输入的 PyTorch 张量（CHW）移动到 CPU 并转换为 NumPy 数组（HWC）
    lm_numpy = (np.transpose(lm_numpy, (0, 2, 3, 1)))# 将数组的维度从 (N, C, H, W) 转换为 (N, H, W, C)，方便后续处理
    nl_list = np.zeros((lm_numpy.shape[0], chn,1))#初始化一个数组，用于存储每张图像（shape(0)）中每个通道(chn)的噪声水平
    for n in range(lm_numpy.shape[0]):# 遍历每张图像
        for c in range(chn):# 遍历每个通道
            # 选择当前图像的当前通道，并重塑为二维数组
            #lm_numpy[n, :, :, c] 返回的是第 n 张图像在第 c 个通道上的所有像素值，
            # 形状为 (H, W)。这个操作通常用于提取特定图像和通道的数据，以便进行后续处理，比如计算噪声水平或应用滤波等。
            #flatten 一个通道  !!reshape操作并不会改变原始数组的数据,它只是返回一个新的数组!!
            selected_lm = np.reshape(lm_numpy[n,:,:,c], (lm_numpy.shape[1]*lm_numpy.shape[2], 1))
            #转成列向量，一个个筛掉小于阈值的数
            # 过滤掉小于阈值的噪声值 大于thre的才留下
            selected_lm = selected_lm[selected_lm>thre]
            # 如果没有满足条件的噪声值，则将噪声水平设置为 0
            if selected_lm.shape[0] == 0:
                nl_list[n, c] = 0
            else: # 计算选定噪声值的直方图，density=True 表示归一化直方图
                #TODO: 直方图
                #从源码对参数a的解释来看，参数a应该传入一维数组以计算直方图。
                # 然而，参数a既可接受PIL.Image对象，也可接受多维数组对象。
                # 参数a之所以能够接受PIL.Image对象，是因为histogram函数内部会通过数组的asarray方法将PIL.Image对象转换为numpy的多维数组；
                # 而参数a之所以能够接受多维数组，是因为histogram函数内部会通过数组的reval方法将多维数组展开成一维数组。

                #将直方图转换为概率密度曲线 返回值第一个是 灰度直方图每个灰度级的频率，也就是灰度直方图的所有纵坐标。
                # 如果由256个灰度级，那么返回值hist的长度便为256。
                hist = np.histogram(selected_lm,  density=True)
                nl_ind = np.argmax(hist[0])# argmax :找到直方图中 最大值的索引
            #print(nl_ind)
            #print(hist[0])
            #print(hist[1])
                nl = ( hist[1][nl_ind] + hist[1][nl_ind+1] ) / 2.# 计算该索引对应的噪声水平
                nl_list[n, c] = nl# 将计算得到的噪声水平存储到 nl_list 中
    return nl_list # 返回所有图像和通道的噪声水平列表

def get_cdf_noise_in_maps(lm, thre=0.8, chn=3):
    '''
    累积分布函数（CDF）
    Description: To find out the most frequent estimated noise level in the images
    描述：找出图像中最常见的估计噪声水平
    ----------
    [Input]
    a multi-channel tensor of noise map
    lm：多通道噪声图的张量
    [Output]
    A list of  noise level value
    噪声水平值的列表
    '''
    lm_numpy = lm.data.cpu().numpy()# 将张量数据移至CPU并转换为NumPy数组
    lm_numpy = (np.transpose(lm_numpy, (0, 2, 3, 1))) # 转置数组，使其形状为 (N, H, W, C)
    nl_list = np.zeros((lm_numpy.shape[0], chn,1))  # 初始化噪声水平列表，形状为 (N, chn, 1)
    for n in range(lm_numpy.shape[0]):
        for c in range(chn):
            # 将当前通道的噪声图展平为一维数组
            selected_lm = np.reshape(lm_numpy[n,:,:,c], (lm_numpy.shape[1]*lm_numpy.shape[2], 1))#展平后 每张图每个通道对应一个值
            # 计算选定噪声图的直方图
            H, x = np.histogram(selected_lm, normed=True)
            dx = x[1]-x[0]
            F = np.cumsum(H)*dx# 计算累积分布函数 (CDF)
            F_ind = np.where(F>thre)[0][0]# 找到 CDF 第一次超过 0.9 的索引
            nl_list[n, c] = x[F_ind]# 存储对应的噪声水平
            print(nl_list[n,c])# 输出噪声水平
    return nl_list

def get_pdf_in_maps(lm, mark, chn):
    '''
    Description: get the noise estimation cdf of each channel
     描述：获取每个通道的噪声估计的概率密度函数（PDF）
    ----------
    [Input]
    a multi-channel tensor of noise map and channel dimension
    chn: the channel number for gaussian
    一个多通道的噪声图张量和通道维度
    chn: 高斯分布的通道数
    [Output]
    CDF function of each sample and each channel
    每个样本和每个通道的 PDF 函数
    '''
    lm_numpy = lm.data.cpu().numpy()
    lm_numpy = (np.transpose(lm_numpy, (0, 2, 3, 1)))
    pdf_list = np.zeros((lm_numpy.shape[0], chn, 10)) #初始化一个数组，用于存储每个图像和每个通道的 PDF
    for n in range(lm_numpy.shape[0]):
        for c in range(chn):
            #选择当前图像的当前通道，并重塑为二维数组
            selected_lm = np.reshape(lm_numpy[n,:,:,c], (lm_numpy.shape[1]*lm_numpy.shape[2], 1))
            #计算选定噪声值的直方图，范围为 [0, 1]，分为 10 个 bins 10个区间 y是每个区间所有的样本数量/频率
            H, x = np.histogram(selected_lm, range=(0.,1.), bins=10, density=True)
            dx = x[1]-x[0]   # 计算每个 bin 的宽度
            F = H * dx  # 计算概率密度函数（PDF）
            pdf_list[n, c, :] = F
            #sio.savemat(mark + str(c) + '.mat',{'F':F})
            plt.bar(range(10), F)# 可视化 PDF，使用柱状图表示
            #plt.savefig(mark + str(c) + '.png')
            plt.close() 
    return pdf_list

def get_pdf_matching_score(F1, F2):
    '''
    Description: Given two sets of CDF, get the overall matching score for each channel
    -----------
    [Input] F1, F2
    [Output] score for each channel
    这个函数的主要目的是评估两个概率密度函数的相似性，适用于比较图像噪声的分布特征，通常在图像处理和计算机视觉任务中使用。
    均方差是一种常见的距离度量，能够有效地反映两个概率分布之间的差异。
    '''
    return np.mean((F1-F2)**2) # np.mean((F1-F2)**2) 计算 F1 和 F2 对应元素之间的平方差，然后取其平均值



def decide_scale_factor(noisy_image, estimation_model, color=1,  thre = 0, plot_flag = 1, stopping = 4, mark=''):
    '''
    Description: Given a noisy image and the noise estimation model, keep multiscaling the image\\
                using pixel-shuffle methods, and estimate the pdf and cdf of AWGN channel
                Compare the changes of the density function and decide the optimal scaling factor

    描述：给定一张噪声图像和噪声估计模型，使用像素重排方法对图像进行多尺度处理，
    估计 AWGN 通道的 PDF 和 CDF，并比较密度函数的变化以决定最佳缩放因子（PD的采样大小）
    ------------
    [Input]  noisy_image, estimation_model, plot_flag, stopping
    [输入]  
    noisy_image：带噪声的图像
    estimation_model：噪声估计模型
    color：指示图像是否为彩色（1 为彩色，0 为灰度）
    thre：匹配得分的阈值
    plot_flag：指示是否绘制图像（默认绘制）
    stopping：最大缩放因子
    mark：用于标记输出文件的字符串

    [Output]  plot the middle vector
            score_seq: the matching score sequence between the two subsequent pdf
            opt_scale: the optimal scaling factor 
    [输出]  
    绘制中间向量
    score_seq：两个连续 PDF 之间的匹配得分序列
    opt_scale：最佳缩放因子 
    '''
    if color == 1:
        c = 3    # 彩色图像有 3 个通道
    elif color == 0:
        c = 1    # 灰度图像有 1 个通道
    score_seq = []   # 用于存储匹配得分的序列
    Pre_CDF = None    # 前一个 PDF
    flag = 0    # 标志，表示是否进行了第一次计算
    
    # 从 1 到 stopping 的循环，用于逐步缩放图像
    for pss in range(1, stopping+1):  #scaling factor from 1 to the limit  缩放因子从 1 到限制
        noisy_image = pixelshuffle(noisy_image, pss)  # 进行像素重排
        INoisy = np2ts(noisy_image, color)  # 将图像转换为张量格式
        #FIXME:INoisy = Variable(INoisy.cuda(), volatile=True)# 将张量转移到 GPU 上
        with torch.no_grad():
            INoisy = Variable(INoisy.cuda())# 将张量转移到 GPU 上
        # 使用估计模型计算噪声图的噪声图
        EMap = torch.clamp(estimation_model(INoisy), 0., 1.) # 获取噪声图的 PDF
        # 计算当前噪声图的 PDF
        EPDF = get_pdf_in_maps(EMap, mark + str(pss), c)[0]
        if flag != 0:# 计算当前 PDF 与前一个 PDF 的匹配得分
            score = get_pdf_matching_score(EPDF, Pre_PDF)  #TODO: How to match these two
            print(score)# 输出匹配得分
            score_seq.append(score) # 将得分加入序列
            if score <= thre:# 如果匹配得分小于等于阈值，返回最佳缩放因子
                print('optimal scale is %d:' % (pss-1))
                return (pss-1, score_seq)    
        Pre_PDF = EPDF # 更新前一个 PDF
        flag = 1 # 标志已进行第一次计算
    return (stopping, score_seq) # 返回最大缩放因子及得分序列

        

def get_max_noise_in_maps(lm, chn=3):
    '''
    Description: To find out the maximum level of noise level in the images
    ----------
    [Input]
    a multi-channel tensor of noise map
    
    [Output]
    A list of  noise level value
    '''
    # 将 lm 转换为 NumPy 数组，并将其转移到 CPU 上
    lm_numpy = lm.data.cpu().numpy()
    # 转置数组的维度，使其形状变为 (batch, height, width, channels) hwc 
    lm_numpy = (np.transpose(lm_numpy, (0, 2, 3, 1)))
    # 初始化存储最大噪声水平的数组
    nl_list = np.zeros((lm_numpy.shape[0], chn, 1))
    for n in range(lm_numpy.shape[0]):
        for c in range(chn):
            nl = np.amax(lm_numpy[n, :, :, c])# 找到该图像的当前通道中噪声图的最大值
            nl_list[n, c] = nl
    return nl_list # 返回每张图像和每个通道的最大噪声水平

def get_smooth_maps(lm, dilk = 50, gsd = 10):
    '''
    Description: To return the refined maps after dilation and gaussian blur
    [Input] a multi-channel tensor of noise map
    [Output] a multi-channel tensor of refined noise map
    描述：该函数 get_smooth_maps 对噪声图像进行膨胀（dilation）和高斯模糊（Gaussian blur）处理，以生成更加平滑的噪声图。
    输入：一个包含噪声图的多通道张量 lm。
    输出：处理后的平滑噪声图张量。
    '''
    kernel = np.ones((dilk, dilk)) # 创建用于图像膨胀的核矩阵
    lm_numpy = lm.data.squeeze(0).cpu().numpy() # 将张量 lm 移至 CPU，去掉 batch 维度，并转换为 NumPy 数组
    lm_numpy = (np.transpose(lm_numpy, (1, 2, 0)))  # 转置数组的维度，使其形状为 (height, width, channels)hwc
    ref_lm_numpy = lm_numpy.copy()  #a refined map  # 复制 lm_numpy，用于存储经过处理的噪声图
    for c in range(lm_numpy.shape[2]):# 遍历每个通道，应用膨胀操作
        nmap = lm_numpy[:, :, c]
        nmap_dilation = cv2.dilate(nmap, kernel, iterations=1)# 对每个通道的噪声图应用膨胀操作
        ref_lm_numpy[:, :, c] = nmap_dilation
        #ref_lm_numpy[:, :, c] = scipy.ndimage.filters.gaussian_filter(nmap_dilation, gsd)
    RF_tensor = np2ts(ref_lm_numpy)# 将处理后的数组转换回张量格式
    RF_tensor = Variable(RF_tensor.cuda(),volatile=True)# 将张量移至 GPU，并设置为不计算梯度

def zeroing_out_maps(lm, keep=0):
    '''
    Only Keep one channel and zero out other channels
    [Input] a multi-channel tensor of noise map
    [Output] a multi-channel tensor of noise map after zeroing out items
    仅保留一个通道，并将其他通道置零
    [输入] 一个多通道的噪声图张量
    [输出] 一个多通道的噪声图张量，将其他通道置零
    '''
    lm_numpy = lm.data.squeeze(0).cpu().numpy()# 将张量 lm 移至 CPU，去掉 batch 维度，并转换为 NumPy 数组
    lm_numpy = (np.transpose(lm_numpy, (1, 2, 0)))# 转置数组的维度，使其形状为 (height, width, channels)
    ref_lm_numpy = lm_numpy.copy()  #a refined map
    for c in range(lm_numpy.shape[2]):
        if np.isin(c,keep)==0:# 如果当前通道不在 `keep` 列表中，将该通道置为零
            ref_lm_numpy[:, :, c] = 0.
    print(ref_lm_numpy)
    RF_tensor = np2ts(ref_lm_numpy)
    RF_tensor = Variable(RF_tensor.cuda(),volatile=True)
    return RF_tensor    #使用循环遍历每个通道，检查当前通道是否需要保留，如果不需要，则将对应通道的所有像素值设为 0。


def level_refine(NM_tensor, ref_mode, chn=3):
    '''
    Description: To refine the estimated noise level maps
    [Input] the noise map tensor, and a refinement mode
    Mode:
    [0] Get the most salient (the most frequent estimated noise level)
    [1] Get the maximum value of noise level
    [2] Gaussian smooth the noise level map to make the regional estimation more smooth
    [3] Get the average maximum value of the noise level
    [5] Get the CDF thresholded value 
    
    [Output] a refined map tensor with four channels
    Description: 细化估计的噪声水平图
    [输入] 噪声图张量和细化模式
    Mode:
    [0] 获取最显著（即频率最高）的噪声水平
    [1] 获取噪声水平的最大值
    [2] 高斯平滑噪声水平图，使区域估计更加平滑
    [3] 获取噪声水平的平均最大值
    [5] 获取累计分布函数（CDF）阈值
    
    [输出] 一个包含四个通道的细化噪声图张量
    '''
    #RF_tensor = NM_tensor.clone()  #get a clone version of NM tensor without changing the original one
    #默认为0
    # 检查ref_mode是否为 0, 1, 4, 或 5，这些模式会对噪声图中的某一特定值进行细化
    if ref_mode == 0 or ref_mode == 1 or ref_mode == 4 or ref_mode==5:  #if we use a single value for the map
        if ref_mode == 0 or ref_mode == 4:# 模式 0 和模式 4：获取噪声图中最显著的噪声水平
            nl_list = get_salient_noise_in_maps(NM_tensor, 0., chn)
            
            if ref_mode == 4:  #half the estimation 如果模式为 4，则将噪声水平减半（相当于置零处理）
                nl_list = nl_list - nl_list
            print(nl_list) # 输出噪声水平列表
        elif ref_mode == 1:# 模式 1：获取噪声图中的最大噪声水平
            nl_list = get_max_noise_in_maps(NM_tensor, chn)
        elif ref_mode == 5: # 模式 5：获取噪声图的累计分布函数阈值
            nl_list = get_cdf_noise_in_maps(NM_tensor, 0.999, chn)
        # 初始化噪声图，将每个通道的所有像素点填充为nl_list中的噪声值
        noise_map = np.zeros((NM_tensor.shape[0], chn, NM_tensor.size()[2], NM_tensor.size()[3]))  #initialize the noise map before concatenating
        for n in range(NM_tensor.shape[0]): # 通过复制nl_list将每个通道填充相同的噪声值，大小为张量的宽和高
            noise_map[n,:,:,:] = np.reshape(np.tile(nl_list[n], NM_tensor.size()[2] * NM_tensor.size()[3]),
                                        (chn, NM_tensor.size()[2], NM_tensor.size()[3]))
        RF_tensor = torch.from_numpy(noise_map).type(torch.FloatTensor)# 将NumPy数组转换为PyTorch张量，并转移到GPU上
        with torch.no_grad():
            RF_tensor = Variable(RF_tensor.cuda())

    elif ref_mode == 2: # 模式 2：应用高斯平滑，调用get_smooth_maps函数
        RF_tensor = get_smooth_maps(NM_tensor, 10, 5)
    elif ref_mode == 3:# 模式 3：计算噪声图中显著值和最大值的平均值
        lb = get_salient_noise_in_maps(NM_tensor)
        up = get_max_noise_in_maps(NM_tensor)
        nl_list = ( lb + up ) * 0.5
        noise_map = np.zeros((1, chn, NM_tensor.size()[2], NM_tensor.size()[3]))  #initialize the noise map before concatenating
        noise_map[0, :, :, :] = np.reshape(np.tile(nl_list, NM_tensor.size()[2] * NM_tensor.size()[3]),
                                        (chn, NM_tensor.size()[2], NM_tensor.size()[3]))
        RF_tensor = torch.from_numpy(noise_map).type(torch.FloatTensor)
        #FIXME: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
        with torch.no_grad():
            RF_tensor = Variable(RF_tensor.cuda())
    

    return (RF_tensor, nl_list) 

def normalize(a, len_v, min_v, max_v):
    '''
    normalize the sequence of factors
     归一化因子序列
    [输入] 
    a: 待归一化的数组或序列
    len_v: 序列的长度（即 a 数组的元素个数）
    min_v: 最小值，用于归一化的下限
    max_v: 最大值，用于归一化的上限

    [输出] 
    norm_a: 归一化后的数组，值范围在 0 到 1 之间

    归一化的意义：
    归一化是数据预处理中的一种常见技术，目的是将数据转换为标准的范围，这样可以：提高模型训练的效率。
    使得不同特征的尺度一致，从而避免某些特征由于尺度较大而主导模型训练过程。
    在这个函数中，使用的归一化方法是 线性归一化，也叫 最小-最大归一化。
    它通过指定的最小值 min_v 和最大值 max_v 将数据映射到 [0, 1] 的范围内。
    '''
    norm_a =  np.reshape(a, (len_v,1))
    norm_a = (norm_a - float(min_v)) / float(max_v - min_v)
    return norm_a

def generate_training_noisy_image(current_image, s_or_m, limit_set, c, val=0):
    noise_level_list = np.zeros((c, 1))
    if s_or_m == 0:  #single noise type    # 如果选择单一噪声类型
        if val == 0:
            for chn in range(c):# 对每个通道生成一个在 limit_set 指定范围内的噪声强度
                noise_level_list[chn] = np.random.uniform(limit_set[0][0], limit_set[0][1])
        elif val == 1:
            for chn in range(c):
                noise_level_list[chn] = 35# 对每个通道设置固定的噪声强度（此处设为 35）
        noisy_img = generate_noisy(current_image, 0, noise_level_list /255.) # 使用生成的噪声强度对当前图像生成带噪声图像
    
    return (noisy_img, noise_level_list)

def generate_ground_truth_noise_map(noise_map, n, noise_level_list, limit_set, c, pn, pw, ph):
    '''   
       生成地面真实噪声图
    [输入]
    noise_map: 一个噪声图张量，用于存储生成的噪声图
    n: 当前噪声图的索引（用于更新对应的噪声图）
    noise_level_list: 噪声强度列表，包含每个通道的噪声强度
    limit_set: 噪声强度范围，用于归一化噪声强度
    c: 图像的通道数
    pn: 图像的批次大小
    pw: 图像的宽度
    ph: 图像的高度

    [输出]
    noise_map: 生成的带有噪声的地面真实噪声图
    '''
    for chn in range(c):# 对噪声强度列表中的每个通道的噪声级别进行归一化
        noise_level_list[chn] = normalize(noise_level_list[chn], 1, limit_set[0][0], limit_set[0][1])  #normalize the level value
    noise_map[n, :, :, :] = np.reshape(np.tile(noise_level_list, pw * ph), (c, pw, ph))  #total number of channels 
    #将噪声级别赋值给每个通道
    return noise_map

#Add noise to the original images
def generate_noisy(image, noise_type, noise_level_list=0, sigma_s=20, sigma_c=40):
    '''
    Description: To generate noisy images of different types
    ----------
    [Input]
    image : ndarray of float type: [0,1] just one image, current support gray or color image input (w,h,c)
    noise_type: 0,1,2,3
    noise_level_list: pre-defined noise level for each channel, without normalization: only information of 3 channels
    [0]'AWGN'     Multi-channel Gaussian-distributed additive noise
    [1]'RVIN'    Replaces random pixels with 0 or 1.  noise_level: ratio of the occupation of the changed pixels
    [2]'Gaussian-Poisson'   GP noise approximator, the combinatin of signal-dependent and signal independent noise
    [Output]
    A noisy image
       Description: 生成不同类型的噪声图像
    ----------
    [输入]
    image : float 类型的 ndarray：[0,1]，表示原始图像（可以是灰度图或彩色图，形状为 [w,h,c]）
    noise_type: 0, 1, 2, 3，表示噪声的类型
    noise_level_list: 每个通道的噪声强度，未进行归一化：只包含三个通道的噪声信息
    [输出]
    A noisy image: 带噪声的图像

    [噪声类型说明]：
    [0] 'AWGN'：多通道高斯加性噪声（AWGN）
    [1] 'RVIN'：替换随机像素为 0 或 1，`noise_level` 表示改变像素的占比
    [2] 'Gaussian-Poisson'：高斯泊松噪声模型，信号相关和非相关噪声的结合
    '''
    w, h, c = image.shape
    #Some unused noise type: Poisson and Uniform
    #if noise_type == *:
        #vals = len(np.unique(image))
        #vals = 2 ** np.ceil(np.log2(vals))
        #noisy = np.random.poisson(image * vals) / float(vals)

    #if noise_type == *:
        #uni = np.random.uniform(-factor,factor,(w, h, c))
        #uni = uni.reshape(w, h, c)
        #noisy = image + uni
    # 复制原始图像，避免修改原始数据
    noisy = image.copy()
    # 噪声类型 0: 高斯噪声（AWGN）
    if noise_type == 0:  #MC-AWGN model
        gauss = np.zeros((w, h, c))# 创建一个与图像相同大小的零矩阵
        for chn in range(c):
            gauss[:,:,chn] = np.random.normal(0, noise_level_list[chn], (w, h)) # 正态分布噪声
        noisy = image + gauss   # 图像加上噪声，得到带噪声的图像
    elif noise_type == 1:  #MC-RVIN model  噪声类型 1: 随机像素噪声（RVIN）
        for chn in range(c):  #process each channel separately # 对每个通道分别处理
            prob_map = np.random.uniform(0.0, 1.0, (w, h))      # 生成随机概率图
            noise_map = np.random.uniform(0.0, 1.0, (w, h))     # 生成随机噪声图
            noisy_chn = noisy[: , :, chn]   # 获取当前通道的图像
            # 根据概率图和噪声图修改当前通道的像素值
            noisy_chn[ prob_map < noise_level_list[chn] ] = noise_map[ prob_map < noise_level_list[chn] ]
            #如果 prob_map 中某个位置的值小于当前通道的噪声水平 noise_level_list[chn]，
            # 则将该位置的像素替换为随机噪声图 noise_map 中对应位置的值。相当于做像素丢失。
    elif noise_type == 2:
        pass #这个模型在代码中没有实现，但可以想象是将高斯噪声和泊松噪声结合的模型。高斯噪声是信号无关的，而泊松噪声则是信号相关的。

    return noisy

#TODO:如何混合噪声？
#generate AWGN-RVIN noise together
def generate_comp_noisy(image, noise_level_list):
    '''
    Description: To generate mixed AWGN and RVIN noise together
    ----------
    [Input]
    image: a float image between [0,1]
    noise_level_list: AWGN and RVIN noise level
    [Output]
    A noisy image
        Description: To generate mixed AWGN and RVIN noise together
    ----------
    [输入]
    image: 一个浮动值范围在 [0, 1] 之间的图像（float 类型的图像）
    noise_level_list: 包含 AWGN 和 RVIN 噪声级别的列表
    [输出]
    A noisy image: 生成的带噪声图像

    在这个列表中：

    noise_level_list[0] 可能是第一个通道的 AWGN 噪声标准差。
    noise_level_list[1] 可能是第二个通道的 AWGN 噪声标准差。
    noise_level_list[2] 可能是第一个通道的 RVIN 混合比例。
    noise_level_list[3] 可能是第二个通道的 RVIN 混合比例。

    '''
    w, h, c = image.shape
    noisy = image.copy()# 复制原始图像，避免修改原始数据
    # 遍历每个通道（假设图像为 RGB 或灰度图）
    for chn in range(c):
        #TODO:怎么就可以通过这个获得混合比例？？？
        # 获取混合噪声的比例（`mix_thre`）和高斯噪声的标准差（`gau_std`）
        # 下面几行是我加的
        print(noise_level_list)
        
        #if noise_level_list != 0.0:
        mix_thre = noise_level_list[c+chn]#只有这行是原来的  #get the mix ratio of AWGN and RVIN 获取 AWGN 和 RVIN 的混合比例
        #elif noise_level_list ==0.0:
        #    mix_thre = 0.5
        
        # 获取 AWGN 和 RVIN 的混合比例 
        gau_std = noise_level_list[chn]  #get the gaussian std 获取当前通道的高斯噪声标准差
        # 生成概率图 `prob_map` 和噪声图 `noise_map`
        prob_map = np.random.uniform( 0, 1, (w, h) ) #the prob map 随机生成一个概率图，值范围为 [0, 1]
        noise_map = np.random.uniform( 0, 1, (w, h) )  #the noisy map 随机生成一个噪声图，用于替换像素
        # 获取当前通道的图像数据
        noisy_chn = noisy[: ,: ,chn]
        # 在当前通道的图像上应用 RVIN 噪声（随机像素替换）
        noisy_chn[prob_map < mix_thre ] = noise_map[prob_map < mix_thre ]
        # 为剩余的像素应用 AWGN（加性高斯噪声）
        gauss = np.random.normal(0, gau_std, (w, h))
        noisy_chn[prob_map >= mix_thre ] = noisy_chn[prob_map >= mix_thre ] + gauss[prob_map >= mix_thre]
        # 对于 prob_map >= mix_thre 的像素（即不被 RVIN 替换的像素），
        # 应用 AWGN（加性高斯噪声）。将 gauss 中的高斯噪声加到 noisy_chn 中。
    return noisy

def generate_denoise(image, model, noise_level_list):
    '''
    Description: Generate Denoised Blur Images
    ----------
     [Input]
    image: 输入的带噪图像，范围是 [0, 1] 之间的浮动值（NumPy 数组）
    model: 用于去噪的深度学习模型
    noise_level_list: 噪声强度列表，包含每个通道的噪声水平，供模型去噪时使用
    
    [Output]
    A blur image patch: 生成的去噪模糊图像
    '''
    #input images
    ISource = np2ts(image) # 使用 np2ts 函数将 NumPy 数组转换为 PyTorch 张量
    ISource = torch.clamp(ISource, 0., 1.)# 确保图像像素值限制在 [0, 1] 范围内
    with torch.no_grad():
        ISource = Variable(ISource.cuda())# 将图像数据移到 GPU 上，并包装成 PyTorch Variable，volatile=True 表示不需要梯度计算
    #input denoise conditions  创建噪声图，用于提供噪声强度信息给模型
     # 初始化一个零矩阵，大小为 (1, 6, height, width)，表示噪声图的尺寸
    noise_map = np.zeros((1, 6, image.shape[0], image.shape[1]))  #initialize the noise map before concatenating
    # 将 noise_level_list 中的噪声强度信息扩展并复制到噪声图中，填充所有像素。
    # 这里将噪声级别信息重复以匹配图像的高度和宽度，并重新调整形状为 (6, height, width)，其中 6 是通道数
    noise_map[0, :, :, :] = np.reshape(np.tile(noise_level_list, image.shape[0] * image.shape[1]), (6, image.shape[0], image.shape[1]))
    

    # 将噪声图转换为 PyTorch 张量并移到 GPU 上
    NM_tensor = torch.from_numpy(noise_map).type(torch.FloatTensor)
    with torch.no_grad():
        NM_tensor = Variable(NM_tensor.cuda())
    #generate blur images  使用模型进行去噪处理
    Res = model(ISource, NM_tensor) # 将输入图像（带噪声）和噪声图传递给去噪模型，得到去噪后的结果
    # 计算去噪图像，`ISource - Res` 表示从原始图像中减去去噪后的结果，从而得到模糊图像
    Out = torch.clamp(ISource-Res, 0., 1.) # 对去噪结果进行阈值处理，确保输出图像的像素值在 [0, 1] 范围内
    # 将输出张量从 GPU 移到 CPU，并转换为 NumPy 数组以便返回
    out_numpy = Out.data.squeeze(0).cpu().numpy()# 使用 `.data` 获取 Tensor 的数据，并去掉批次维度（squeeze(0)），然后移到 CPU 并转换为 NumPy 数组
    out_numpy = np.transpose(out_numpy, (1, 2, 0)) # 将 NumPy 数组的维度从 (channels, height, width) 转换为 (height, width, channels)，符合图像格式
    
    return out_numpy  


#TODO: two pixel shuffle functions to process the images
def pixelshuffle(image, scale):
    '''
    Discription: Given an image, return a reversible sub-sampling
    [Input]: Image ndarray float
    [Return]: A mosic image of shuffled pixels
    描述：给定一张图像，返回可逆的子采样图像
    [输入]：图像 ndarray（浮点型）
    [返回]：一个像素打乱后的马赛克图像

    切片语法的解释
    ws::scale 表示从第 ws 行开始，每隔 scale 行取一个元素。
    hs::scale 表示从第 hs 列开始，每隔 scale 列取一个元素。
    : 表示选择所有的通道。
    WHC格式的image

    axis=0: 表示在第一个轴（行）上进行连接，即将数组纵向叠加。
    axis=1：表示在第二个轴（列）上进行连接，即将数组横向并排放置。
    如果提供的数组是多维的，选择不同的轴将影响最终的形状。

    注意： temp = image[ws::scale, hs::scale, :]里 
    是隔到scale个就选  例如[[1,2,3],[4,5,6],[7,8,9]]
    隔2个得到[[1,3],[7,9]]  隔一个是不变的
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
            
def reverse_pixelshuffle(image, scale, fill=0, fill_image=0, ind=[0,0]):
    '''
    Discription: Given a mosaic image of subsampling, recombine it to a full image
    [Input]: Image
    [Return]: Recombine it using different portions of pixels
    描述: 给定一个经过子采样的马赛克图像，将其重建为完整图像
    [输入]: image - 输入的马赛克图像
    [返回]: 通过不同像素部分重组的完整图像
    功能: 这个函数接受一个经过像素打乱处理的马赛克图像，并根据给定的缩放因子将其重建为原始图像。
    '''
    w, h, c = image.shape
    real = np.zeros((w, h, c))  #real image 创建一个与输入图像同样大小的全零数组 real，用于存储重建后的图像。
    wf = 0# 写入的行索引
    hf = 0# 写入的列索引
    for ws in range(scale):# 外层循环遍历行缩放因子
        hf = 0# 每次进入新行时重置列索引
        for hs in range(scale):# 内层循环遍历列缩放因子
            temp = real[ws::scale, hs::scale, :]# 提取当前区域的像素到临时变量
            wc, hc, cc = temp.shape  #get the shpae of the current images
            if fill==1 and ws==ind[0] and hs==ind[1]:# 如果需要填充并且是特定索引
                real[ws::scale, hs::scale, :] = fill_image[wf:wf+wc, hf:hf+hc, :] # 用填充图像填充当前区域
            else:
                real[ws::scale, hs::scale, :] = image[wf:wf+wc, hf:hf+hc, :] # 否则，从输入的马赛克图像中提取数据填充
            hf = hf + hc# 更新列索引，准备写入下一个位置
        wf = wf + wc# 更新行索引，准备进入下一行的循环
    return real # 返回重建后的完整图像
        
def scal2map(level, h, w,  min_v=0., max_v=255.):
    '''
    Change a single normalized noise level value to a map
    [Input]: level: a scaler noise level(0-1), h, w
    [Return]: a pytorch tensor of the cacatenated noise level map
    
    将单一的归一化噪声级别值转换为噪声图
    [输入]: 
    level: 一个标量噪声级别（范围是 0 到 1），h 和 w 分别表示图像的高度和宽度
    [返回]: 
    一个包含拼接噪声级别的 PyTorch 张量
    '''
    #get a tensor from the input level 将输入的标量噪声级别 'level' 转换为 PyTorch 张量，并调整形状为 (1, 1)
    level_tensor = torch.from_numpy(np.reshape(level, (1,1))).type(torch.FloatTensor)
    #make the noise level to a map 通过将 'level_tensor' 重塑为 (1, 1, 1, 1)，为后续扩展到整个图像做准备
    level_tensor = level_tensor.view(stdN_tensor.size(0), stdN_tensor.size(1), 1, 1)
    # 将噪声级别 'level' 扩展到图像的所有像素位置，生成一个大小为 (1, 1, h, w) 的噪声图
    level_tensor = level_tensor.repeat(1, 1, h, w)
    return level_tensor

def scal2map_spatial(level1, level2, h, w):
    # 生成上半部分的噪声图（噪声级别为 'level1'）
    stdN_t1 = scal2map(level1, int(h/2), w)
    # 生成下半部分的噪声图（噪声级别为 'level2'）
    stdN_t2 = scal2map(level2, h-int(h/2), w)
    # 在高度方向（dim=2）拼接上半部分和下半部分的噪声图，形成完整的噪声图
    stdN_tensor = torch.cat([stdN_t1, stdN_t2], dim=2)
    return stdN_tensor


#TODO:数组切片
'''
NumPy 中的数组切片是左闭右开的。这意味着在使用切片语法时，包含起始索引的元素，但不包含结束索引的元素。例如：
array[start:end] 会返回从 start 到 end-1 的元素。
如果 start 是 0，end 是 5，那么切片结果将包含索引 0 到 4 的元素。

在使用 [start:end] 进行切片时，如果 end 超过了数组的边界，不会报错，而是会自动调整为数组的实际边界。
也就是说，切片操作会返回从 start 到数组的末尾的元素。
这种切片方式在编程中是常见的，尤其是在 Python 中，这样设计可以避免许多边界问题，使得索引计算更加直观。

你说得对。在使用 array[start::stride] 的切片时，从 start 开始，每隔 stride 个元素取一个，直到数组的末尾。
这种方式并不改变左闭右开的特性，只是通过 stride 设定了取样的间隔。
例如，对于一个长度为 10 的数组 array，使用 array[1::2] 会返回索引 1、3、5、7 和 9 的元素。
这里的 start 是 1，而 stride 是 2，因此选择的是从索引 1 开始每隔 2 个元素的元素。
如果 start 加上 stride 超出了数组的边界，切片将会自动停止，不会抛出错误。这样设计使得处理数组的切片操作更加灵活和安全。
'''