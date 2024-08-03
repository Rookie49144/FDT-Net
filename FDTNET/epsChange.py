import os
from PIL import Image
import matplotlib.pyplot as plt

def convert_images_to_eps(input_folder, output_folder):
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入目录中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            try:
                # 打开图像
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path)

                # 转换为RGB模式（避免透明度问题）
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                # 创建输出文件路径
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, f"{base_name}.eps")

                # 使用Matplotlib保存为EPS
                plt.figure()
                plt.imshow(img)
                plt.axis('off')  # 不显示坐标轴
                plt.savefig(output_path, format='eps', bbox_inches='tight', pad_inches=0)
                plt.close()

                print(f"Converted {filename} to EPS format.")
            except Exception as e:
                print(f"Error converting {filename}: {e}")

if __name__ == "__main__":
    input_folder = r'C:\Users\Rookie\Desktop\low_contrast\image'
    output_folder = r'C:\Users\Rookie\Desktop\low_contrast\eps'
    convert_images_to_eps(input_folder, output_folder)