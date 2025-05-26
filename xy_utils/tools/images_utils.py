import os
from PIL import Image
import json
import re
def get_first_image_info(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return
    
    # 获取文件夹中所有文件
    files = os.listdir(folder_path)
    
    # 查找第一个图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_path = None
    
    for file in files:
        file_extension = os.path.splitext(file)[1].lower()
        if file_extension in image_extensions:
            image_path = os.path.join(folder_path, file)
            break
    
    # 如果没有找到图片文件
    if image_path is None:
        print(f"错误：在文件夹 '{folder_path}' 中未找到图片文件")
        return
    
    try:
        # 打开图片并获取信息
        with Image.open(image_path) as img:
            width, height = img.size
            channels = len(img.getbands())
            
            print(f"找到图片：{os.path.basename(image_path)}")
            print(f"分辨率：{width} x {height} 像素")
            print(f"通道数：{channels}")
            if channels == 1:
                print("通道说明：单通道（可能是灰度图）")
            elif channels == 3:
                print("通道说明：三通道（RGB）")
            elif channels == 4:
                print("通道说明：四通道（RGBA，包含Alpha通道）")
            else:
                print(f"通道说明：非常规通道数（{channels}通道）")
                
    except Exception as e:
        print(f"错误：无法处理图片 '{image_path}' - {str(e)}")

def generate_depth_params_json(png_folder, output_path):
    """
    遍历指定文件夹中的所有 PNG 图像，提取文件名并生成 depth_params.json 文件。
    
    参数:
    - png_folder: PNG 图像所在的文件夹路径
    - output_path: 生成的 JSON 文件的保存路径
    """
    # 检查 PNG 文件夹是否存在
    if not os.path.exists(png_folder):
        print(f"错误：PNG 图像文件夹 '{png_folder}' 不存在")
        return
    
    # 获取所有 PNG 文件的文件名（不包含扩展名）
    png_files = [f for f in os.listdir(png_folder) 
                if os.path.isfile(os.path.join(png_folder, f)) 
                and f.lower().endswith('.png')]
    
    # 提取文件名（不含扩展名）
    base_names = [os.path.splitext(f)[0] for f in png_files]
    
    # 如果没有找到 PNG 文件
    if not base_names:
        print(f"错误：在文件夹 '{png_folder}' 中未找到 PNG 文件")
        return
    
    # 构建 JSON 数据
    json_data = {}
    for name in base_names:
        json_data[name] = {
            "scale": 0.0,
            "offset": 0.0
        }
    
    # 创建输出文件所在的目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 写入 JSON 文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        print(f"成功生成 JSON 文件：{output_path}")
        print(f"共处理 {len(base_names)} 个 PNG 文件")
    except Exception as e:
        print(f"错误：无法写入 JSON 文件 '{output_path}' - {str(e)}")
        
def resize_image(input_path, output_path=None,scale=2, quality=100):
    """
    input_path = '/home/qinllgroup/hongxiangyu/git_project/livo2-data-utils/10-Mesh-acc/data/delete/stack_acc_10_2/00142-final.png'
    output_path = '/home/qinllgroup/hongxiangyu/git_project/gaussian-splatting-xy/data/tree_01_save_w_depth/depth_maps_2/00142.png'
    resize_image(input_path, output_path)
    
    将图片分辨率缩小1/2
    
    参数:
    - input_path: 输入图片路径
    - output_path: 输出图片路径，默认为在原文件名后加 '_resized'
    - quality: 输出图片质量，范围0-100，默认为95
    """
    try:
        # 打开图片
        with Image.open(input_path) as img:
            # 获取原始尺寸
            width, height = img.size
            
            # 计算新尺寸（缩小1/2）
            new_width = width // scale
            new_height = height // scale
            
            # 使用高质量重采样方法
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 如果没有指定输出路径，自动生成
            if output_path is None:
                base, ext = os.path.splitext(input_path)
                output_path = f"{base}_resized{ext}"
            
            # 保存图片，保持原始格式
            resized_img.save(output_path, quality=quality)
            
            print(f"成功将图片从 {width}x{height} 缩小到 {new_width}x{new_height}")
            print(f"保存路径: {output_path}")
            
            return output_path
            
    except Exception as e:
        print(f"错误: 无法处理图片 {input_path} - {str(e)}")
        return None
    
def batch_rename_files(folder_path):
    """
    批量重命名指定文件夹中的所有文件，移除文件名末尾的 -final
    
    参数:
    - folder_path: 要处理的文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return
    
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    
    # 用于匹配 -final 的正则表达式模式
    pattern = re.compile(r'^(.*?)-final(\.[^.]+)?$')
    
    renamed_count = 0
    
    # 遍历所有文件并进行重命名
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        
        # 只处理文件，不处理文件夹
        if os.path.isfile(file_path):
            # 使用正则表达式匹配文件名
            match = pattern.match(filename)
            
            if match:
                # 获取新文件名
                new_name = match.group(1) + (match.group(2) or '')
                new_path = os.path.join(folder_path, new_name)
                
                try:
                    # 执行重命名
                    os.rename(file_path, new_path)
                    print(f"已重命名: {filename} -> {new_name}")
                    renamed_count += 1
                except Exception as e:
                    print(f"错误：无法重命名文件 '{filename}' - {str(e)}")
    
    print(f"重命名完成！共处理 {renamed_count} 个文件")