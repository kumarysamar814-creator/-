import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

os.makedirs('output_images', exist_ok=True)
img = cv2.imread(r"C:\Users\25909\PyCharmMiscProject\preview.jpg")

#输出原图片
(b, g, r) = img[100, 100]
print(b, g, r)
plt.imshow(img)
plt.title('Original Image')
plt.savefig('output_images/original_bgr.jpg', dpi=300)
plt.show()

#蓝绿色翻转
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)
plt.title('RGB Image')
plt.savefig('output_images/rgb_image.jpg', dpi=300)
plt.show()

#转换为灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray')
plt.title('Gray Image')
plt.savefig('output_images/gray_image.jpg', dpi=300)
plt.show()


# ==================== 添加噪声 ====================

# 添加椒盐噪声
sp_noise_img = random_noise(
    rgb_img,
    mode='s&p',
    amount=0.4
)

# 添加高斯噪声
gus_noise_img = random_noise(
    rgb_img,
    mode='gaussian',
    mean=0.2,
    var=0.03
)

# 转换为 0~255 的 uint8 格式
sp_uint8 = (sp_noise_img * 255).astype(np.uint8)
gus_uint8 = (gus_noise_img * 255).astype(np.uint8)


# ==================== 均值滤波 ====================

mean_sp = cv2.blur(sp_uint8, (5, 5))
mean_gus = cv2.blur(gus_uint8, (5, 5))


# ==================== 中值滤波 ====================

mid_sp = cv2.medianBlur(sp_uint8, 5)
mid_gus = cv2.medianBlur(gus_uint8, 5)


# ==================== 高斯滤波 ====================

gauss_sp = cv2.GaussianBlur(sp_uint8, (5, 5), 0)
gauss_gus = cv2.GaussianBlur(gus_uint8, (5, 5), 0)


# ==================== 展示滤波结果 ====================

plt.figure(figsize=(13, 8))

# 第一行：椒盐噪声的滤波结果
plt.subplot(2, 3, 1)
plt.imshow(mean_sp)
plt.title('S&P Noise - Mean Filter')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(mid_sp)
plt.title('S&P Noise - Median Filter')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(gauss_sp)
plt.title('S&P Noise - Gaussian Filter')
plt.axis('off')

# 第二行：高斯噪声的滤波结果
plt.subplot(2, 3, 4)
plt.imshow(mean_gus)
plt.title('Gaussian Noise - Mean Filter')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(mid_gus)
plt.title('Gaussian Noise - Median Filter')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(gauss_gus)
plt.title('Gaussian Noise - Gaussian Filter')
plt.axis('off')

plt.tight_layout()
plt.savefig('output_images/filter_results.jpg', dpi=300)
plt.show()

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise


# ==================== 1. 读取原始图片 ====================

image_path = r"C:\Users\25909\PyCharmMiscProject\preview.jpg"

# 创建结果保存文件夹
output_dir = os.path.join(
    os.path.dirname(image_path),
    "output_images"
)
os.makedirs(output_dir, exist_ok=True)

# 读取图片
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(
        f"图片读取失败，请检查文件路径：{image_path}"
    )

# BGR 转换为 RGB
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("原始图片读取成功")
print("原始图片尺寸：", rgb_img.shape)


# ==================== 2. 缩小图片，避免手动滤波过慢 ====================

MAX_SIDE = 600

h, w = rgb_img.shape[:2]

if max(h, w) > MAX_SIDE:
    scale = MAX_SIDE / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    rgb_img = cv2.resize(
        rgb_img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    print("图片已缩小为：", rgb_img.shape)

h, w = rgb_img.shape[:2]


# ==================== 3. 添加椒盐噪声 ====================

sp_noise_img = random_noise(
    rgb_img,
    mode='s&p',
    amount=0.4
)

# 转换为 0~255 的 uint8 格式
sp_uint8 = (
    np.clip(sp_noise_img, 0, 1) * 255
).astype(np.uint8)

print("椒盐噪声添加完成")


# ==================== 4. 手动实现中值滤波 ====================

def manual_median_filter_color(image, kernel_size=5):
    """
    手动实现彩色图像中值滤波。

    image: 输入的 RGB 彩色图像
    kernel_size: 滤波窗口大小，必须是正奇数
    return: 手动中值滤波后的图像
    """

    # 检查参数
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("滤波窗口大小必须是正奇数")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("输入必须是三通道彩色图像")

    # 窗口半径：5×5 窗口对应半径 2
    pad = kernel_size // 2

    # 创建输出图像
    filtered_img = np.zeros_like(image)

    # 对图像边缘进行填充，避免边缘像素无法处理
    padded_img = np.pad(
        image,
        pad_width=((pad, pad), (pad, pad), (0, 0)),
        mode='edge'
    )

    # 遍历每一行
    for i in range(image.shape[0]):

        # 输出处理进度
        if i % 50 == 0:
            print(f"手动滤波进度：{i}/{image.shape[0]} 行")

        # 遍历每一列
        for j in range(image.shape[1]):

            # 取出当前像素周围的窗口
            region = padded_img[
                i:i + kernel_size,
                j:j + kernel_size,
                :
            ]

            # 分别计算 RGB 三个通道的中位数
            median_value = np.median(
                region,
                axis=(0, 1)
            )

            # 将中值写入输出图像
            filtered_img[i, j] = median_value.astype(np.uint8)

    return filtered_img


print("开始手动中值滤波，请稍候……")

manual_mid = manual_median_filter_color(
    sp_uint8,
    kernel_size=5
)

print("手动中值滤波完成！")


# ==================== 5. 显示处理结果 ====================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(rgb_img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sp_uint8)
plt.title("Salt & Pepper Noise")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(manual_mid)
plt.title("Manual Median Filter")
plt.axis('off')

plt.tight_layout()

# 保存处理结果
save_path = os.path.join(
    output_dir,
    "manual_median_comparison.jpg"
)

plt.savefig(save_path, dpi=300)

print("结果图片已保存到：", save_path)

# 最后显示窗口
plt.show()