# Encoding: UTF-8
import os
from itertools import chain
from pathlib import Path
import csv


def listdir_img(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


if __name__ == "__main__":
    # print(os.getcwd())
    if os.path.exists('src'):
        os.chdir('src')
    # files = listdir_img('../stargan-v2/profile_photo/Import')
    os.chdir('../read/')
    files = os.listdir('./')
    files.sort()
    row = []
    with open('../重命名结果.csv', 'w', newline='', encoding='utf-8')as f:
        f_csv = csv.writer(f)
        for i in range(len(files)):
            # os.path.splitext()  # 分离文件名与扩展名
            # os.path.splitext(file)[0]  # 获得文件名
            # os.path.splitext(file)[1]  # 获得文件扩展名
            row = [os.path.splitext(files[i])[0] + os.path.splitext(files[i])[1],
                   'pixiv_%s' % i + os.path.splitext(files[i])[1]]
            f_csv.writerow(row)
            os.rename(files[i], 'pixiv_%s' % i + os.path.splitext(files[i])[1])
