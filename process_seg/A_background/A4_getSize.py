import numpy as np
import cv2

from mmseg.apis import init_model, inference_model, show_result_pyplot

import matplotlib.pyplot as plt


# img_path = 'data/street_uk.jpeg'

def show_img_info(img_path):
    img_bgr = cv2.imread(img_path)
    print(img_bgr.shape)
    plt.imshow(img_bgr[:, :, ::-1])
    plt.show()

img_path = 'E:\\Square\\mmsegmentation\\Square_Semantic_Seg_Mask\\ann_dir\\train\\Square_1.png'
show_img_info(img_path)
img_path = 'E:\\Square\\mmsegmentation\\Square_Semantic_Seg_Mask\\img_dir\\train\\Square_1.jpg'
show_img_info(img_path)



# RuntimeError: stack expects each tensor to be equal size, but got [3, 1024, 1593] at entry 0 and [3, 1024, 1225] at entry 1