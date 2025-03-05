import json
import cv2
import os

# 设定文件夹路径，包含所有JSON标注文件和图像
json_folder = 'annotations/'
image_folder = 'images/'

# 目标文件夹，保存裁剪后的结果
output_json_folder = 'output_annotations/'
output_image_folder = 'output_images/'

# 确保输出文件夹存在
os.makedirs(output_json_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)

# 目标尺寸
target_width, target_height = 640, 480

# 获取所有JSON文件
json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

for json_file in json_files:
    json_path = os.path.join(json_folder, json_file)
    image_path = os.path.join(image_folder, json_file.replace('.json', '.jpg'))  # 假设图像文件和JSON文件同名，扩展名不同
    
    # 加载标注文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 原图尺寸
    original_width, original_height = 1920, 1440

    # 获取所有标注物体的边界框
    objects = data['objects']
    bboxes = [(obj['bndbox'][0], obj['bndbox'][1], 
               obj['bndbox'][0] + obj['bndbox'][2], 
               obj['bndbox'][1] + obj['bndbox'][3]) for obj in objects]

    # 计算包含所有物体的最小矩形
    min_x = min(bbox[0] for bbox in bboxes)
    min_y = min(bbox[1] for bbox in bboxes)
    max_x = max(bbox[2] for bbox in bboxes)
    max_y = max(bbox[3] for bbox in bboxes)

    # 计算包含矩形的中心
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # 计算裁剪框的左上角，使裁剪框的中心尽量接近包含矩形的中心
    crop_x = max(0, int(center_x - target_width / 2))
    crop_y = max(0, int(center_y - target_height / 2))

    # 确保裁剪框不会超出图像边界
    crop_x = min(crop_x, original_width - target_width)
    crop_y = min(crop_y, original_height - target_height)

    # 裁剪范围
    crop_box = (crop_x, crop_y, crop_x + target_width, crop_y + target_height)

    # 更新标注信息
    updated_objects = []
    for obj in objects:
        x, y, w, h = obj['bndbox']
        x_new = x - crop_x
        y_new = y - crop_y
        
        # 计算裁剪后坐标
        if 0 <= x_new < target_width and 0 <= y_new < target_height:
            obj['bndbox'] = [x_new, y_new, w, h]
            updated_objects.append(obj)

    # 更新JSON文件
    data['objects'] = updated_objects
    data['image_shape'] = [target_height, target_width, 1]
    
    # 保存裁剪后的JSON文件
    output_json_path = os.path.join(output_json_folder, json_file)
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)

    # 裁剪图像
    image = cv2.imread(image_path)
    cropped_image = image[crop_y:crop_y + target_height, crop_x:crop_x + target_width]
    
    # 保存裁剪后的图像
    output_image_path = os.path.join(output_image_folder, json_file.replace('.json', '.jpg'))
    cv2.imwrite(output_image_path, cropped_image)

print("批处理完成！")
