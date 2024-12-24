import os
os.chdir('E:\Square\mmsegmentation')
print(os.getcwd())

# 创建 checkpoint 文件夹，用于存放预训练模型权重文件
os.mkdir('checkpoint')

# 创建 outputs 文件夹，用于存放预测结果
os.mkdir('outputs')

# 创建 data 文件夹，用于存放图片和视频素材
os.mkdir('data')

# 创建 图表 文件夹，用于存放生成的图表
os.mkdir('graphs')

# 创建 Zihao-Configs 文件夹，用于存放自己的语义分割模型的 config 配置文件
os.mkdir('MyConfigs')