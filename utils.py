import os
import random
import numpy as np
import torch
from args_fusion import args
from scipy.misc import imread, imsave, imresize
import matplotlib as mpl
import pdb
import cv2

from os import listdir
from os.path import join

EPSILON = 1e-5

###### Testing ########
'''
def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        elif name.endswith('.tif'):
            images.append(join(directory, file))
        # name1 = name.split('.')
        names.append(name)
    return images, names

###### Training ########
'''
def list_images(directory):
    print(directory)
    images = []
    names = []
    dir = os.listdir(directory)
    dir.sort()
    for dir0 in dir:
        #print(directory, dir0)
        #print(os.listdir(os.path.join(directory, dir0)).sort())
        for dir1 in os.listdir(os.path.join(directory, dir0)):
            req_path = os.path.join(directory, dir0, dir1, 'lwir')
            for file in os.listdir(req_path):
                name = file
                if name.endswith('.png'):
                    images.append(join(req_path, file))
                elif name.endswith('.jpg'):
                    images.append(join(req_path, file))
                elif name.endswith('.jpeg'):
                    images.append(join(req_path, file))
                elif name.endswith('.bmp'):
                    images.append(join(req_path, file))
                elif name.endswith('.tif'):
                    images.append(join(req_path, file))
                names.append(name)
                #print(join(req_path, file), name)
    return images, names

# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def get_image(path, height=256, width=256, flag=False):
    if flag is True:
        image = imread(path, mode='RGB')
    else:
        image = imread(path, mode='L')

    if height is not None and width is not None:
        image = imresize(image, [height, width], interp='bicubic')
    return image


# load images - test phase
def get_test_image(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        if flag is True:
            image = imread(path, mode='RGB')
        else:
            image = imread(path, mode='L')
        # get saliency part
        if height is not None and width is not None:
            #image = imresize(image, [height, width], interp='nearest')
            image = imresize(image, [height, width], interp='bicubic')

        base_size = 512
        h = image.shape[0]
        w = image.shape[1]
        c = 1
        if h > base_size or w > base_size:
            c = 4
            if flag is True:
                image = np.transpose(image, (2, 0, 1))
            else:
                image = np.reshape(image, [1, h, w])
            images = get_img_parts(image, h, w)
        else:
            if flag is True:
                image = np.transpose(image, (2, 0, 1))
            else:
                image = np.reshape(image, [1, image.shape[0], image.shape[1]])
            images.append(image)
            images = np.stack(images, axis=0)
            images = torch.from_numpy(images).float()

    return images, h, w, c


def get_img_parts(image, h, w):
    images = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    img1 = image[:, 0:h_cen + 3, 0: w_cen + 3]
    img1 = np.reshape(img1, [1, img1.shape[0], img1.shape[1], img1.shape[2]])
    img2 = image[:, 0:h_cen + 3, w_cen - 2: w]
    img2 = np.reshape(img2, [1, img2.shape[0], img2.shape[1], img2.shape[2]])
    img3 = image[:, h_cen - 2:h, 0: w_cen + 3]
    img3 = np.reshape(img3, [1, img3.shape[0], img3.shape[1], img3.shape[2]])
    img4 = image[:, h_cen - 2:h, w_cen - 2: w]
    img4 = np.reshape(img4, [1, img4.shape[0], img4.shape[1], img4.shape[2]])
    images.append(torch.from_numpy(img1).float())
    images.append(torch.from_numpy(img2).float())
    images.append(torch.from_numpy(img3).float())
    images.append(torch.from_numpy(img4).float())
    return images

def recons_fusion_images(img_lists, h, w):
    img_f_list = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    c = img_lists[0][0].shape[1]
    ones_temp = torch.ones(1, c, h, w).cuda()
    for i in range(len(img_lists[0])):
        # img1, img2, img3, img4
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]

        img_f = torch.zeros(1, c, h, w).cuda()
        count = torch.zeros(1, c, h, w).cuda()

        img_f[:, :, 0:h_cen + 3, 0: w_cen + 3] += img1
        count[:, :, 0:h_cen + 3, 0: w_cen + 3] += ones_temp[:, :, 0:h_cen + 3, 0: w_cen + 3]
        img_f[:, :, 0:h_cen + 3, w_cen - 2: w] += img2
        count[:, :, 0:h_cen + 3, w_cen - 2: w] += ones_temp[:, :, 0:h_cen + 3, w_cen - 2: w]
        img_f[:, :, h_cen - 2:h, 0: w_cen + 3] += img3
        count[:, :, h_cen - 2:h, 0: w_cen + 3] += ones_temp[:, :, h_cen - 2:h, 0: w_cen + 3]
        img_f[:, :, h_cen - 2:h, w_cen - 2: w] += img4
        count[:, :, h_cen - 2:h, w_cen - 2: w] += ones_temp[:, :, h_cen - 2:h, w_cen - 2: w]
        img_f = img_f / count
        img_f_list.append(img_f)
    return img_f_list

def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion) + EPSILON)
    img_fusion = img_fusion * 255
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    img_fusion = cv2.resize(img_fusion, (256,256), interpolation = cv2.INTER_CUBIC)
    imsave(output_path, img_fusion)


def get_train_images(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, height, width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images
