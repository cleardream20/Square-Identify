import os
import json
import numpy as np
import cv2

import matplotlib.pyplot as plt

# os.chdir("C:\\Users\\26685\\Desktop\\Squares")
# print(os.getcwd())


img_path = 'C:\\Users\\26685\\Desktop\\Squares\\Square_5.png'
img_bgr = cv2.imread(img_path)
print(img_bgr.shape)

plt.imshow(img_bgr[:,:,::-1])
plt.show()

img_mask = np.zeros(img_bgr.shape[:2])
print(img_mask)

plt.imshow(img_mask)
plt.show()

labelme_json_path = 'C:\\Users\\26685\\Desktop\\Squares\\Square_5.json'
with open(labelme_json_path, 'r', encoding='utf-8') as f:
    labelme = json.load(f)
print(labelme.keys())

# 元数据
print(labelme['version'])
print(labelme['imagePath'])
print(labelme['imageHeight'])
print(labelme['imageWidth'])

class_info = [
    {'label':'Space', 'type':'polygon', 'color':1},
    {'label':'Square', 'type':'polygon', 'color':2}
]

for one_class in class_info:  # 按顺序遍历每一个类别
    for each in labelme['shapes']:  # 遍历所有标注，找到属于当前类别的标注
        if each['label'] == one_class['label']:
            if one_class['type'] == 'polygon':  # polygon 多段线标注

                # 获取点的坐标
                points = [np.array(each['points'], dtype=np.int32).reshape((-1, 1, 2))]

                # 在空白图上画 mask（闭合区域）
                img_mask = cv2.fillPoly(img_mask, points, color=one_class['color'])

            elif one_class['type'] == 'line' or one_class['type'] == 'linestrip':  # line 或者 linestrip 线段标注

                # 获取点的坐标
                points = [np.array(each['points'], dtype=np.int32).reshape((-1, 1, 2))]

                # 在空白图上画 mask（非闭合区域）
                img_mask = cv2.polylines(img_mask, points, isClosed=False, color=one_class['color'],
                                         thickness=one_class['thickness'])

            elif one_class['type'] == 'circle':  # circle 圆形标注

                points = np.array(each['points'], dtype=np.int32)

                center_x, center_y = points[0][0], points[0][1]  # 圆心点坐标

                edge_x, edge_y = points[1][0], points[1][1]  # 圆周点坐标

                radius = np.linalg.norm(np.array([center_x, center_y] - np.array([edge_x, edge_y]))).astype(
                    'uint32')  # 半径

                img_mask = cv2.circle(img_mask, (center_x, center_y), radius, one_class['color'],
                                      one_class['thickness'])

            else:
                print('未知标注类型', one_class['type'])

plt.imshow(img_mask)
plt.show()

print(img_mask)
mask_path = img_path.split('.')[0] + '.png'
cv2.imwrite(mask_path, img_mask)

mask_img = cv2.imread('C:\\Users\\26685\\Desktop\\Squares\\Square_5.png')
plt.imshow(mask_img[:,:,0])
plt.show()