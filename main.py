# Encoding: UTF-8
import argparse
import numpy as np
import torch
import cv2
import os
from src.model import SketchKeras
import src.image_cutter as img_ctr
import src.image_stitcher as img_str
import random
from tqdm import tqdm

# import copy
# import glob
# import time

device = "cuda" if torch.cuda.is_available() else "cpu"


# def cv_imread(file_path):
#     cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
#     return cv_img
#
#
# def cv_imwrite(filename, src):
#     cv2.imencode('.jpg', src)[1].tofile(filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str, default="read/", help="input images"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="save/", help="output images"
    )
    parser.add_argument(
        "--weight", "-w", type=str, default="weights/model.pth", help="weight file"
    )
    parser.add_argument(
        '--ratio', type=float, default=0.95,
        help='how many of the pictures will be used as train data other pics will be used as val'
    )
    parser.add_argument(
        '--size_limit', type=int, default=1536,
        help='if the resolution of the image is too high, ' +
             'the image will be resized to a proper size around this number'
    )
    return parser.parse_args()


def preprocess(img):
    h, w, c = img.shape
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    highpass = img.astype(int) - blurred.astype(int)
    highpass = highpass.astype(np.float) / 128.0

    # 假如出现黑块的话原本是这里导致的，因为np.max(highpass)，即除数为0了导致程序错误
    tmp = np.max(highpass)
    if tmp == 0:
        tmp += 1
    highpass /= tmp

    ret = np.zeros((512, 512, 3), dtype=np.float)
    ret[0:h, 0:w, 0:c] = highpass
    return ret


def postprocess(pred, thresh=0.18, smooth=False):
    assert 1.0 >= thresh >= 0.0

    pred = np.amax(pred, 0)
    pred[pred < thresh] = 0
    pred = 1 - pred
    pred *= 255
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    if smooth:
        pred = cv2.medianBlur(pred, 3)
    return pred


def resize_img(image, size):
    biggest = image.shape[0] if image.shape[0] > image.shape[1] else image.shape[1]
    k = size / biggest
    image = cv2.resize(image, (int(image.shape[1] * k), int(image.shape[0] * k)))
    return image


def dir_check():
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if not os.path.exists(args.output + '/val'):
        os.mkdir(args.output + '/val')
    if not os.path.exists(args.output + '/train'):
        os.mkdir(args.output + '/train')
    if not os.path.exists(args.output + '/' + 'train' + '/' + 'color'):
        os.makedirs(args.output + '/' + 'train' + '/' + 'color')
    if not os.path.exists(args.output + '/' + 'train' + '/' + 'gray'):
        os.makedirs(args.output + '/' + 'train' + '/' 'gray')
    if not os.path.exists(args.output + '/' + 'train' + '/' 'sketch'):
        os.makedirs(args.output + '/' + 'train' + '/' 'sketch')
    if not os.path.exists(args.output + '/' + 'val' + '/' + 'color'):
        os.makedirs(args.output + '/' + 'val' + '/' + 'color')
    if not os.path.exists(args.output + '/' + 'val' + '/' + 'gray'):
        os.makedirs(args.output + '/' + 'val' + '/' 'gray')
    if not os.path.exists(args.output + '/' + 'val' + '/' 'sketch'):
        os.makedirs(args.output + '/' + 'val' + '/' 'sketch')


if __name__ == "__main__":
    args = parse_args()
    model = SketchKeras().to(device)
    dir_check()

    names = os.listdir(args.input)
    names2 = os.listdir(args.output + 'train/sketch/') + os.listdir(args.output + 'val/sketch/')

    total_number = len(names)
    total_train = 0
    total_val = 0
    # random.shuffle(names)

    # names.sort()
    # names2.sort()

    if len(args.weight) > 0:
        model.load_state_dict(torch.load(args.weight))
        print(f"{args.weight} loaded..")

    for num_of_img in tqdm(range(len(names))):
        name = names[num_of_img]
        if name not in names2:
            try:
                img = cv2.imread(args.input + '/' + name)
                if (args.size_limit < img.shape[0]) or (args.size_limit < img.shape[1]):
                    img = resize_img(img, args.size_limit)
                img_height = img.shape[0]
                img_width = img.shape[1]
            except AttributeError:
                print(name + ' 读取失败')
            else:
                train_not_val = random.random() < args.ratio
                # 彩图处理部分
                if train_not_val:
                    cv2.imwrite(
                        args.output + '/train/color/' + name,
                        img
                    )
                else:
                    cv2.imwrite(
                        args.output + '/val/color/' + name,
                        img
                    )
                # 黑白处理部分
                if train_not_val:
                    cv2.imwrite(
                        args.output + '/train/gray/' + name,
                        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    )
                else:
                    cv2.imwrite(
                        args.output + '/val/gray/' + name,
                        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    )
                # 线稿处理部分
                part, num_width, num_height = img_ctr.main(img)
                new_part = np.zeros((num_height - 1, num_width - 1, 512, 512), dtype=np.uint8)
                for i in range(num_height - 1):
                    for j in range(num_width - 1):
                        img = part[i][j]
                        # preprocess
                        img = preprocess(img)
                        x = img.reshape(1, *img.shape).transpose(3, 0, 1, 2)
                        x = torch.tensor(x).float()

                        # feed into the network
                        with torch.no_grad():
                            pred = model(x.to(device))
                        pred = pred.squeeze()

                        # postprocess
                        output = pred.cpu().detach().numpy()
                        output = postprocess(output, thresh=0.1, smooth=False)

                        new_part[i][j] = output
                        '''# cv2.namedWindow('test', cv2.WINDOW_NORMAL)
                        # cv2.imshow('test', new_part[i][j])
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        # input()

                        # cv2.imwrite(
                        #     args.output + '/' + '%s%s_' % (i, j) + name,
                        #     new_part[i][j]
                        # )'''

                new_img = np.zeros((256 * num_height, 256 * num_width), dtype=np.uint8)
                '''下面逻辑比较复杂，调好了就别瞎鸡儿动了 之所以i是-1，j是-2是因为组合rows比组合每一行的part晚一步，
                因为part已经全有了，但是跑一轮才能有第一个row。所以处理part和row的逻辑不统一，这里可以改但没必要'''
                # 从上到下合并每行
                for i in range(num_height - 1):
                    new_row = np.zeros((512, 256 * num_width), dtype=np.uint8)
                    # 先把这一行中的第一幅图片粘贴到应有的位置准备就绪
                    # new_row[0:512, 0:512] = new_part[i][0]
                    # 从左到右合并一行中的每张图片
                    for j in range(num_width - 2):
                        if j == 0:
                            new_row[0:512, 0:((j + 3) * 256)] = \
                                img_str.main(
                                    new_part[i][0],
                                    new_part[i][j + 1],
                                    True
                                )
                            '''cv2.namedWindow('0+1', cv2.WINDOW_NORMAL)
                            cv2.imshow('0+1', new_row[0:512, 0:((j + 3) * 256)])
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                            input()'''
                        else:
                            # 很坑爹，必须用deepcopy不然不是复制一份数据而是复制的索引！！！（然而之后发现问题似乎并不是出在这里
                            # tmp = copy.deepcopy(new_row[0:512, 0:((j + 2) * 256)])
                            new_row[0:512, 0:((j + 3) * 256)] = \
                                img_str.main(
                                    new_row[0:512, 0:((j + 2) * 256)],
                                    new_part[i][j + 1],
                                    True
                                )
                        '''# cv2.imwrite(
                        #     args.output + '/' + '%s' % j + name,
                        #     new_row[0:512, 0:((j + 3) * 256)]
                        # )
                    # cv2.namedWindow('a full row', cv2.WINDOW_NORMAL)
                    # cv2.imshow('a full row', new_row)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # input()'''
                    if i == 0:
                        new_img[0:512, 0:256 * num_width] = new_row
                    else:
                        # tmp = copy.deepcopy(new_img[0:512 + 256 * (i - 1), 0:256 * num_width])
                        new_img[0:512 + 256 * i, 0:256 * num_width] = \
                            img_str.main(
                                new_img[0:512 + 256 * (i - 1), 0:256 * num_width],
                                new_row,
                                False
                            )
                '''# cv2.namedWindow('a full img', cv2.WINDOW_NORMAL)
                # cv2.imshow('a full img', new_img[256:256 + img_height, 256:256 + img_width])
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # input()'''
                if train_not_val:
                    cv2.imwrite(
                        args.output + '/train/sketch/' + name,
                        new_img[256:256 + img_height, 256:256 + img_width]
                    )
                    print(name + '处理完成，为train')
                    total_train += 1
                else:
                    cv2.imwrite(
                        args.output + '/val/sketch/' + name,
                        new_img[256:256 + img_height, 256:256 + img_width]
                    )
                    print(name + '处理完成，为val')
                    total_val += 1

    print('全部', total_number, '张图片处理完成，其中', total_train, '张为train，', total_val, '张为val')
