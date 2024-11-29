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
parser.add_argument("--ps", type=int, default=1, help='pixel shuffle [0]no pixel-shuffle [1]adaptive pixel-ps [2]pre-set stride')
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

def img_normalize(data):
    return data/255.

def main():

    seismic_noise = read_segy_data('D:\\academic\\dncnn_pytorch\\data\\sgy_data\\synthetic.sgy')  # 野外地震数据        
    seismic_block_h, seismic_block_w = seismic_noise.shape
    # 数据归一化处理
    seismic_noise_max = abs(seismic_noise).max()  # 获取数据最大幅值
    seismic_noise = seismic_noise / seismic_noise_max  # 将数据归一化到(-1,1)  
    seismic_noise = seismic_noise[:, :,np.newaxis] 

    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
    # Build model
    print('Loading model ...\n')
    c = 1 if opt.color == 0 else 3
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

    # load data info
    print('Loading data info ...\n')
    #TODO:相对先改绝对 后面要看怎么改
    #files_source = glob.glob(os.path.join('data', opt.test_data, '*.*'))、
    
    #files_source = glob.glob(os.path.join("D:\\academic\\PD-Denoising-pytorch\\data\\beijing", '*.*'))
    #files_source.sort()
    #t = 0
    #process images with pre-defined noise level
    #for f in files_source:
        #t = t+1
    #print(f)
    #file_name = f.split('/')[-1].split('.')[0]
    #print(file_name)
    # if opt.real_n == 2:  #have ground truth
    #     gnd_file_path = os.path.join('data',opt.test_data_gnd, file_name + '_mean.png')
    #     print(gnd_file_path)
    #     Img_gnd = cv2.imread(gnd_file_path)
    #     Img_gnd = Img_gnd[:,:,::-1]
    #     Img_gnd = cv2.resize(Img_gnd, (0,0), fx=opt.scale, fy=opt.scale)
    #     Img_gnd = img_normalize(np.float32(Img_gnd))
    # # image
    #Img = cv2.imread(f)  #input image with w*h*c
    Img = seismic_noise

    pss=1
    if opt.ps == 1:
        #TODO:相对先改绝对 后面要看怎么改
        #pss = decide_scale_factor(Img/255., model_est, color=opt.color,  thre = 0.008, plot_flag = 1, stopping = 4,mark = opt.out_dir + '/' +  file_name)[0]
        pss = decide_scale_factor(Img/seismic_noise_max, model_est, color=opt.color,  thre = 0.008, plot_flag = 1, stopping = 4,mark = "D:\\academic\\PD-Denoising-pytorch\\results\\beijing" + '/' + "seis")[0]
        print(pss)
        Img = pixelshuffle(Img, pss)
    elif opt.ps == 2:
        pss = opt.ps_scale
    
    merge_out= np.zeros([w,h,3])
    print('Splitting and Testing.....')
    wbin = opt.wbin
    i = 0
    while i < w:
        i_end = min(i+wbin, w)
        j = 0
        while j < h:
            j_end = min(j+wbin, h)
            patch = Img[i:i_end,j:j_end,:]
            patch_merge_out_numpy = denoiser(patch, c, pss, model, model_est, opt)
            merge_out[i:i_end, j:j_end, :] = patch_merge_out_numpy        
            j = j_end
        i = i_end
    #TODO:相对先改绝对 后面要看怎么改
    #FIXME:图片逆pd
    out_image = reverse_pixelshuffle(merge_out[:,:,::-1],pss)
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
