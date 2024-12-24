import os
os.chdir('E:\Square\mmsegmentation')

import numpy as np
import cv2

from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv

import matplotlib.pyplot as plt

# 模型 config 配置文件
config_file = 'configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py'

# 模型 checkpoint 权重文件
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'

model = init_model(config_file, checkpoint_file, device='cuda:0')

img_path = 'data/street_uk.jpeg'
img_bgr = cv2.imread(img_path)
print(img_bgr.shape)
plt.imshow(img_bgr[:,:,::-1])
plt.show()

result = inference_model(model, img_bgr)
print(result)
print(result.keys())

# 一些参数
print(result.pred_sem_seg.data.shape)
print(np.unique(result.pred_sem_seg.data.cpu()))
print(result.pred_sem_seg.data.shape)
print(result.pred_sem_seg.data)

pred_mask = result.pred_sem_seg.data[0].detach().cpu().numpy()
plt.imshow(pred_mask)
plt.show()

# 定量置信度
print(result.seg_logits.data.shape)


from mmseg.datasets import cityscapes
# import numpy as np
import mmcv
from PIL import Image

# 获取类别名和调色板
classes = cityscapes.CityscapesDataset.METAINFO['classes']
palette = cityscapes.CityscapesDataset.METAINFO['palette']
opacity = 0.15 # 透明度，越大越接近原图

# 将分割图按调色板染色
# seg_map = result[0].astype('uint8')
seg_map = pred_mask.astype('uint8')
seg_img = Image.fromarray(seg_map).convert('P')
seg_img.putpalette(np.array(palette, dtype=np.uint8))

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
plt.figure(figsize=(14, 8))
im = plt.imshow(((np.array(seg_img.convert('RGB')))*(1-opacity) + mmcv.imread(img_path)*opacity) / 255)

# 为每一种颜色创建一个图例
patches = [mpatches.Patch(color=np.array(palette[i])/255., label=classes[i]) for i in range(len(classes))]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')

# plt.savefig('outputs/B2-4.jpg')
plt.show()