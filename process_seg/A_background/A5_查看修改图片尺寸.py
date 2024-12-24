from pprint import pformat

from PIL import Image

pnum = '6'
ploc = 'ann'
pform = 'png'
pt_v = 'val'

img_path = f'E:\\Square\\mmsegmentation\\Square_Semantic_Seg_Mask\\{ploc}_dir\\{pt_v}\\Square_{pnum}.{pform}'
save_path = f'E:\\Square\\mmsegmentation\\Square_Semantic_Seg_Mask\\{ploc}_dir\\{pt_v}\\Square_{pnum}.{pform}'

def get_size(img_path):
    img = Image.open(img_path)

    width, height = img.size
    print(f'Width: {width}, Height: {height}')


"""
设定宽度
"""
def fix_size_wd(img_path, save_path):
    # 打开一个图片文件
    img = Image.open(img_path)

    # 定义新的尺寸（宽度，高度）
    new_width = 2048
    new_height = 1024

    # 调整图片尺寸
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 保存修改后的图片
    resized_img.save(save_path)


"""
保持纵横比
"""
def fix_size_scale(img_path, save_path):
    # 打开一个图片文件
    img = Image.open(img_path)

    # 获取图片的原始尺寸
    original_width, original_height = img.size

    # 定义新的宽度
    new_width = 800

    # 计算新的高度，保持纵横比
    aspect_ratio = original_height / original_width
    new_height = int(new_width * aspect_ratio)

    # 调整图片尺寸
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 保存修改后的图片
    resized_img.save(save_path)


fix_size_wd(img_path, save_path)
get_size(img_path)
