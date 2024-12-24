import os
from enum import unique

os.chdir('E:\Square\mmsegmentation')
print(os.getcwd())

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt

# 指定单张图像路径
img_path = 'Square_Semantic_Seg_Mask/img_dir/train/Square_1.jpg'
mask_path = 'Square_Semantic_Seg_Mask/ann_dir/train/Square_1.png'

img = cv2.imread(img_path)
mask = cv2.imread(mask_path)

print(img.shape)
print(mask.shape)

# mask灰度图标注含义
print(np.unique(mask))
plt.figure(figsize=(10, 6))
plt.imshow(mask*50)
plt.axis('off')
plt.show()

# 每个类别的 BGR 配色
# palette = [
#     ['background', [127,127,127]],
#     ['red', [0,0,200]],
#     ['green', [0,200,0]],
#     ['white', [144,238,144]],
#     ['seed-black', [30,30,30]],
#     ['seed-white', [8,189,251]]
# ]

palette = [
    ['background', [127,127,127]],
    ['square', [200,0,0]],
    ['space', [0,0,200]]
]

# 红-square
# 蓝-space

palette_dict = {}
for idx, each in enumerate(palette):
    palette_dict[idx] = each[1]

print(palette_dict)

mask = mask[:,:,0]

# 将整数ID，映射为对应类别的颜色
viz_mask_bgr = np.zeros((mask.shape[0], mask.shape[1], 3))
for idx in palette_dict.keys():
    viz_mask_bgr[np.where(mask==idx)] = palette_dict[idx]
viz_mask_bgr = viz_mask_bgr.astype('uint8')

# 将语义分割标注图和原图叠加显示
opacity = 0.2 # 透明度越大，可视化效果越接近原图
label_viz = cv2.addWeighted(img, opacity, viz_mask_bgr, 1-opacity, 0)

plt.figure(figsize=(10, 6))
plt.imshow(label_viz[:,:,::-1])
plt.axis('off')
plt.show()

cv2.imwrite('outputs/D-1.jpg', label_viz)

def batch_proc(PATH_IMAGE, PATH_MASKS):
    # n 行 n 列可视化
    n = 5

    # 透明度越大，可视化效果越接近原图
    opacity = 0.2

    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(16, 12))

    for i, file_name in enumerate(os.listdir(PATH_IMAGE)[:n ** 2]):

        # 载入图像和标注
        img_path = os.path.join(PATH_IMAGE, file_name)
        mask_path = os.path.join(PATH_MASKS, file_name.split('.')[0] + '.png')
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        mask = mask[:, :, 0]

        # 将预测的整数ID，映射为对应类别的颜色
        viz_mask_bgr = np.zeros((mask.shape[0], mask.shape[1], 3))
        for idx in palette_dict.keys():
            viz_mask_bgr[np.where(mask == idx)] = palette_dict[idx]
        viz_mask_bgr = viz_mask_bgr.astype('uint8')

        # 将语义分割标注图和原图叠加显示
        label_viz = cv2.addWeighted(img, opacity, viz_mask_bgr, 1 - opacity, 0)

        # 可视化
        axes[i // n, i % n].imshow(label_viz[:, :, ::-1])
        axes[i // n, i % n].axis('off')  # 关闭坐标轴显示
    fig.suptitle('Image and Semantic Label', fontsize=30)
    # plt.tight_layout()
    plt.savefig('outputs/D-2.jpg')
    plt.show()

path_i = 'Square_Semantic_Seg_Mask/img_dir/train'
path_m = 'Square_Semantic_Seg_Mask/ann_dir/train'
batch_proc(path_i, path_m)
