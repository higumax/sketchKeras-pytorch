import cv2
import os
import numpy as np


def how_many(length):
    return int(length / 256) + 2


def extend_img(img, img_width, img_height, num_width, num_height):
    return cv2.copyMakeBorder(img, 256, num_height  * 256 - img_height,
                              256, num_width * 256 - img_width,
                              borderType=cv2.BORDER_REFLECT)  # 反射法


def cut_img(img, num_width, num_height):
    part = np.zeros((num_height - 1, num_width - 1, 512, 512, 3), dtype=np.uint8)
    # 第i行第j个，从左到右从上到下
    for i in range(num_height - 1):
        for j in range(num_width - 1):
            part[i][j] = img[(i * 256):(i * 256 + 512), (j * 256):(j * 256 + 512)]
            name = '%s%s%s' % (i, ' ', j)
            # print(name + '裁剪完成')
    return part


# def save_cutted_img(part, num_width, num_height):
#     for i in range(num_width):
#         for j in range(num_height):
#             cv2.imwrite((i, '_', j, '.png'), part[i][j])


def main(img):
    # 计算一下按照256*256扩充长宽各需要几块
    img_height = img.shape[0]
    img_width = img.shape[1]
    num_width = how_many(img_width)
    num_height = how_many(img_height)

    reflect = extend_img(img, img_width, img_height, num_width, num_height)
    part = cut_img(reflect, num_width, num_height)
    return part, num_width, num_height
    # reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,
    #                              borderType=cv2.BORDER_REFLECT)  # 反射法

    # dst1 = img[0:512, 0:512]   # 裁剪坐标为[y0:y1, x0:x1]
    # cv2.namedWindow('img_color', cv2.WINDOW_NORMAL)
    # cv2.imshow('img_color', reflect)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#
# if __name__ == '__main__':
#     if os.path.exists('test'):
#         os.chdir('test')
#     image = cv2.imread('1.png', cv2.IMREAD_COLOR)
#     main(image)
