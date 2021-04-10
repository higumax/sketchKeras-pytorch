import cv2
import numpy as np
import os


def calWeight(d, k):
    '''
    :param d: 融合重叠部分直径
    :param k: 融合计算权重参数
    :return:
    '''

    x = np.arange(-d / 2, d / 2)
    y = 1 / (1 + np.exp(-k * x))
    return y


def imgFusion(img1, img2, overlap, left_right):
    '''
    图像加权融合
    :param img1:
    :param img2:
    :param overlap: 重合长度
    :param left_right: 是否是左右融合
    :return:
    '''
    # 这里先暂时考虑平行向融合
    w = calWeight(overlap, 0.05)  # k=5 这里是超参

    if left_right:  # 左右融合
        row1, col1 = img1.shape
        row2, col2 = img2.shape
        img_new = np.zeros((row1, col1 + col2 - overlap))
        img_new[0:row1, 0:col1] = img1
        w_expand = np.tile(w, (row1, 1))  # 权重扩增
        img_new[0:row1, (col1 - overlap):col1] = \
            (1 - w_expand) * img1[0:row1, (col1 - overlap):col1] + \
            w_expand * img2[0:row2, 0:overlap]
        img_new[:, col1:] = img2[:, overlap:]
    else:  # 上下融合
        row1, col1 = img1.shape
        row2, col2 = img2.shape
        img_new = np.zeros((row1 + row2 - overlap, col1))
        img_new[0:row1, 0:col1] = img1
        w = np.reshape(w, (overlap, 1))
        w_expand = np.tile(w, (1, col1))
        img_new[row1 - overlap:row1, 0:col1] = \
            (1 - w_expand) * img1[(row1 - overlap):row1, 0:col1] + \
            w_expand * img2[0:overlap, 0:col2]
        img_new[row1:, :] = img2[overlap:, :]
    return img_new


def main(img1, img2, left_right):
    img1 = (img1 - img1.min()) / img1.ptp()
    img2 = (img2 - img2.min()) / img2.ptp()
    result = imgFusion(img1, img2, 256, left_right)
    # result = np.uint16(result * 65535)
    return np.uint8(result * 255)


# if __name__ =="__main__":
#     img1 = cv2.imread('save/00_1.png', cv2.IMREAD_UNCHANGED)
#     img2 = cv2.imread('save/01_1.png', cv2.IMREAD_UNCHANGED)
#     img1 = (img1 - img1.min()) / img1.ptp()
#     img2 = (img2 - img2.min()) / img2.ptp()
#     result = imgFusion(img1, img2, overlap=256, left_right=True)
#     result = np.uint16(result * 65535)
#     cv2.imwrite('save/01手动.png', result)

#     # cv2.namedWindow('test', cv2.WINDOW_NORMAL)
#     # cv2.imshow('test', result)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

# 加权平均融合法 https://blog.csdn.net/xiaoxifei/article/details/103045958
