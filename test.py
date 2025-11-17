import numpy as np
from PIL import Image

# 目标图像尺寸
width = 960
height = 256

# 降低分辨率（生成更粗糙的噪声）
scale_factor = 1  # 颗粒度大小（值越大，颗粒越大）
low_width = width // scale_factor
low_height = height // scale_factor

# 生成低分辨率随机噪声
low_res_noise = np.random.rand(low_height, low_width)  # 0 到 1 之间的随机值
low_res_noise = (low_res_noise * 255).astype(np.uint8)  # 转换为 0-255 的整数

# 放大到目标尺寸
image = Image.fromarray(low_res_noise, mode='L')  # 创建低分辨率图像
image = image.resize((width, height), Image.NEAREST)  # 使用最近邻插值放大

# 保存图像
output_path = "coarse_noise_image.png"
image.save(output_path)

print(f"颗粒度较大的噪声图像已保存到: {output_path}")