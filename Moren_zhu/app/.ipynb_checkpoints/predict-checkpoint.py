# -*- coding: utf-8 -*-
# Author:  @ MuLun_Zhu
# reader > programmer > machine
# @Time :  2021/6/8 8:36 上午

import os
import joblib
import unet
from glob import glob
import torch
import warnings
from tqdm import tqdm
import numpy as np
from skimage.io import imread, imsave
import matplotlib
import matplotlib.pyplot as plt
import cv2
from skimage.io import imsave

args = joblib.load('./args.pkl')
joblib.dump(args, './args.pkl')

def crop_ceter(img,croph,cropw):
    #for n_slice in range(img.shape[0]):
    height,width = img[0].shape
    starth = height//2-(croph//2)
    startw = width//2-(cropw//2)
    return img[:,starth:starth+croph,startw:startw+cropw]

def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    #有点像“去掉最低分去掉最高分”的意思,使得数据集更加“公平”
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)#限定范围numpy.clip(a, a_min, a_max, out=None)

    #除了黑色背景外的区域要进行标准化
    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9 #黑色背景区域
        return tmp

def predicting(filepath,filename):
    model = unet.__dict__[args.arch](args)
    model = model.cuda()
    data1 = np.load(filepath,allow_pickle=True)
    print(data1.shape)
    print(data1[:, :, 0].shape)
    
    data2 = data1[:, :, 0]
    data2 = np.where(data2<-8, -2.67, data2)
    data2 = data2+2.67
#     matlplotlib
    print(data2.max())
    print(data2.min())
#     fig, ax = plt.subplots(figsize=plt.figaspect(data2))
#     ax.set_xticks()
#     ax.set_yticks()
#     left, bottom, right, top, wspace, hspace
#     fig.subplots_adjust(0, 0, 1, 1)
#     ax.imshow(data2, interpolation='none', cmap='gray')
#     fig.savefig("static/images/output/{name}".format(name=filename))
    
    imsave("static/images/upload/{name}".format(name=filename),data2)
#     data1 = normalize(data1) # 标准化
#     data1 = crop_ceter(data1,160,160)
    data1 = data1.T
    data1 = np.expand_dims(data1,0)
    #data1.reshape()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_ = torch.from_numpy(data1).type(torch.FloatTensor)
    img = img_.to(device=device, dtype=torch.float32)
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()

    with torch.no_grad():
        img = img.cuda()
        output = model(img)
        # print("img_paths[i]:%s" % img_paths[i])
        output = torch.sigmoid(output).data.cpu().numpy()
        print(output.shape)
        np.save('test.npy', output)

    for i in range(output.shape[0]):
        """
        生成灰色圖片
        wtName = os.path.basename(img_paths[i])
        overNum = wtName.find(".npy")
        wtName = wtName[0:overNum]
        wtName = wtName + "_WT" + ".png"
        imsave('output/%s/'%args.name + wtName, (output[i,0,:,:]*255).astype('uint8'))
        tcName = os.path.basename(img_paths[i])
        overNum = tcName.find(".npy")
        tcName = tcName[0:overNum]
        tcName = tcName + "_TC" + ".png"
        imsave('output/%s/'%args.name + tcName, (output[i,1,:,:]*255).astype('uint8'))
        etName = os.path.basename(img_paths[i])
        overNum = etName.find(".npy")
        etName = etName[0:overNum]
        etName = etName + "_ET" + ".png"
        imsave('output/%s/'%args.name + etName, (output[i,2,:,:]*255).astype('uint8'))
        """
        rgbPic = np.zeros([160, 160, 3], dtype=np.uint8)
        for idx in range(output.shape[2]):
            for idy in range(output.shape[3]):
                if output[i,0,idx,idy] > 0.5:
                    rgbPic[idx, idy, 0] = 0
                    rgbPic[idx, idy, 1] = 128
                    rgbPic[idx, idy, 2] = 0
                if output[i,1,idx,idy] > 0.5:
                    rgbPic[idx, idy, 0] = 255
                    rgbPic[idx, idy, 1] = 0
                    rgbPic[idx, idy, 2] = 0
                if output[i,2,idx,idy] > 0.5:
                    rgbPic[idx, idy, 0] = 255
                    rgbPic[idx, idy, 1] = 255
                    rgbPic[idx, idy, 2] = 0
        rgbPic = np.array(rgbPic)
        print(rgbPic.shape)
        for i in range(3):
            rgbPic[:,:,i]=rgbPic[:,:,i].T
        imsave('static/images/output/{name}'.format(name=filename),rgbPic)
