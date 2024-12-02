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
from PIL import Image
import scipy.io as sio

#the limitation range of each type of noise level: [0]Gaussian [1]Impulse
limit_set = [[0,75], [0, 80]]

def segy_normalize(seismic_noise):
    seismic_noise_max = abs(seismic_noise).max()  # 获取数据最大幅值
    seismic_noise = seismic_noise / seismic_noise_max  # 将数据归一化到(-1,1)
    return seismic_noise

def img_normalize(data):
    return data/255.

def denoiser(Img, c, pss, model, model_est, opt):
    with torch.no_grad():
        w, h, _ = Img.shape   
        Img = pixelshuffle(Img, pss)

        #FIXME:选择图像或者是segy 当前：segy
        #Img = img_normalize(np.float32(Img))
        seismic_noise_max = abs(Img).max()
        Img = segy_normalize(np.float32(Img))
        

        noise_level_list = np.zeros((2 * c,1))  #初始化噪声级别列表，2*c 表示双重噪声类型，每个通道有不同噪声
        if opt.cond == 0:  #使用噪声的真实值（ground truth）进行去噪，只有一个噪声类型
            noise_level_list = np.array(opt.test_noise_level)
        elif opt.cond == 2:  # 使用外部固定输入条件进行去噪
            noise_level_list = np.array(opt.ext_test_noise_level)
        
        #将图像(whc)转换为张量(cwh)，以便后续处理
        ISource = np2ts(Img) 
        # noisy image and true residual
        if opt.real_n == 0 and opt.spat_n == 0:  #没有空间噪声设置，使用合成噪声
            #TODO:这里是不是先生成两种类型噪声再合并
            print(opt.test_noise_level)
            #FIXME:选择图像或者是segy 当前：segy
            #noisy_img = generate_comp_noisy(Img, np.array(opt.test_noise_level) / 255.)
            noisy_img = generate_comp_noisy(Img, np.array(opt.test_noise_level) / seismic_noise_max)
            if opt.color == 0:
                noisy_img = np.expand_dims(noisy_img[:,:,0], 2)  # 如果是灰度图像，提取第一个通道 成为whc
        elif opt.real_n == 1 or opt.real_n == 2:  #testing real noisy images测试真实的非合成带噪图像
            noisy_img = Img    # 直接使用输入图像作为带噪图像
        elif opt.spat_n == 1:  # 使用空间噪声
            noisy_img = generate_noisy(Img, 2, 0, 20, 40)# 生成空间噪声（泊松） 没有实现！
        INoisy = np2ts(noisy_img, opt.color)
        
        #FIXME:我不应该clamp，注意图片和sgy的区别
        INoisy = torch.clamp(INoisy, 0., 1.)
        
        
        True_Res = INoisy - ISource   #得到真实的残差
        #FIXME: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
        #ISource, INoisy, True_Res = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True), Variable(True_Res.cuda(),volatile=True)
        ISource, INoisy, True_Res = Variable(ISource.cuda()), Variable(INoisy.cuda()), Variable(True_Res.cuda())

        if opt.mode == "MC": # 如果模式是 "MC"（多条件）
            # 如果使用实际的噪声级别（ground truth）或固定的噪声级别
            if opt.cond == 0 or opt.cond == 2:  #if we use ground choose level or the given fixed level
                #normalize noise leve map to [0,1]
                
                noise_level_list_n = np.zeros((2*c, 1))
                print(c)
                for noise_type in range(2):
                    for chn in range(c):
                        noise_level_list_n[noise_type * c + chn] = normalize(noise_level_list[noise_type * 3 + chn], 1, limit_set[noise_type][0], limit_set[noise_type][1])  #normalize the level value
                #generate noise maps 生成噪声图
                noise_map = np.zeros((1, 2 * c, Img.shape[0], Img.shape[1]))  #initialize the noise map
                noise_map[0, :, :, :] = np.reshape(np.tile(noise_level_list_n, Img.shape[0] * Img.shape[1]), (2*c, Img.shape[0], Img.shape[1]))
                NM_tensor = torch.from_numpy(noise_map).type(torch.FloatTensor)
                  #FIXME: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
        #ISource, INoisy, True_Res = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True), Variable(True_Res.cuda(),volatile=True)
                ISource, INoisy, True_Res = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True), Variable(True_Res.cuda(),volatile=True)
                NM_tensor = Variable(NM_tensor.cuda())
            #use the estimated noise-level map for blind denoising
            elif opt.cond == 1:  #if we use the estimated map directly
                #FIXME:我不应该clamp，注意图片和sgy的区别
               # NM_tensor = torch.clamp(model_est(INoisy), 0., 1.)#使用强度预估模型
                NM_tensor = model_est(INoisy)#使用强度预估模型
                
                if opt.refine == 1:  #if we need to refine the map before putting it to the denoiser
                    NM_tensor_bundle = level_refine(NM_tensor, opt.refine_opt, 2*c)  #refine_opt can be max, freq and their average
                    NM_tensor = NM_tensor_bundle[0]
                    noise_estimation_table = np.reshape(NM_tensor_bundle[1], (2 * c,))
                if opt.zeroout == 1:
                    NM_tensor = zeroing_out_maps(NM_tensor, opt.keep_ind)
            Res = model(INoisy, NM_tensor)

        elif opt.mode == "B":
            Res = model(INoisy)
        #FIXME:我不应该clamp，注意图片和sgy的区别
        #Out = torch.clamp(INoisy-Res, 0., 1.)  #Output image after denoising
        Out = INoisy-Res
        
        
        #FIXME:标准化的数值 自行改变
        #get the maximum denoising result
        max_NM_tensor = level_refine(NM_tensor, 1, 2*c)[0]
        max_Res = model(INoisy, max_NM_tensor)
         #FIXME:我不应该clamp，注意图片和sgy的区别
        #max_Out = torch.clamp(INoisy - max_Res, 0., 1.)
        max_Out = INoisy - max_Res
        
        max_out_numpy = visual_va2np(max_Out, opt.color, opt.ps, pss, 1, opt.rescale, w, h, c)
        max_out_numpy = max_out_numpy/255*seismic_noise_max
        del max_Out
        del max_Res
        del max_NM_tensor
        
        if (opt.ps == 1 or opt.ps == 2) and pss!=1:  #pixelshuffle multi-scale
            #create batch of images with one subsitution
            mosaic_den = visual_va2np(Out, opt.color, 1, pss, 1, opt.rescale, w, h, c)
            out_numpy = np.zeros((pss ** 2, c, w, h))
            #compute all the images in the ps scale set
            for row in range(pss):
                for column in range(pss):
                    re_test = visual_va2np(Out, opt.color, 1, pss, 1, opt.rescale, w, h, c, 1, visual_va2np(INoisy, opt.color), [row, column])/255.
                    #cv2.imwrite(os.path.join(opt.out_dir,file_name + '_%d_%d.png' % (row, column)), re_test[:,:,::-1]*255.)
                    re_test = np.expand_dims(re_test, 0)
                    if opt.color == 0:  #if gray image
                        re_test = np.expand_dims(re_test[:, :, :, 0], 3)
                    re_test_tensor = torch.from_numpy(np.transpose(re_test, (0,3,1,2))).type(torch.FloatTensor)
                    #FIXME: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
                    re_test_tensor = Variable(re_test_tensor.cuda())
                             #FIXME:我不应该clamp，注意图片和sgy的区别
                    #re_NM_tensor = torch.clamp(model_est(re_test_tensor), 0., 1.)
                    re_NM_tensor = model_est(re_test_tensor)
                    
                    if opt.refine == 1:  #if we need to refine the map before putting it to the denoiser
                            re_NM_tensor_bundle = level_refine(re_NM_tensor, opt.refine_opt, 2*c)  #refine_opt can be max, freq and their average
                            re_NM_tensor = re_NM_tensor_bundle[0]
                    re_Res = model(re_test_tensor, re_NM_tensor)
                             #FIXME:我不应该clamp，注意图片和sgy的区别
                    #Out2 = torch.clamp(re_test_tensor - re_Res, 0., 1.)
                    Out2 = re_test_tensor - re_Res

                    out_numpy[row*pss+column,:,:,:] = Out2.data.cpu().numpy()
                    del Out2
                    del re_Res
                    del re_test_tensor
                    del re_NM_tensor
                    del re_test
                
            out_numpy = np.mean(out_numpy, 0)
            #FIXME:选择图像或者是segy 当前：segy
            #out_numpy = np.transpose(out_numpy, (1,2,0)) * 255.0
            out_numpy = np.transpose(out_numpy, (1,2,0))*seismic_noise_max
        elif opt.ps == 0 or pss==1:  #other cases
            out_numpy = visual_va2np(Out, opt.color, 0, 1, 1, opt.rescale, w, h, c)
            out_numpy = out_numpy/255 * seismic_noise_max
        out_numpy = out_numpy.astype(np.float32)  #details
        max_out_numpy = max_out_numpy.astype(np.float32)  #background

        #merging the details and background to balance the effect
        k = opt.k
        merge_out_numpy = (1-k)*out_numpy + k*max_out_numpy
        merge_out_numpy = merge_out_numpy.astype(np.float32)
        
        return merge_out_numpy


