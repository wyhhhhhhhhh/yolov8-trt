import json
import numpy as np
import cv2
import os

# 设置图像和JSON文件的目录
image_dir = "./images"  # 图像目录
json_dir = "./annotations"   # JSON标注目录
output_dir = "./croped_txt"  # 输出标注目录

os.makedirs(output_dir, exist_ok=True)

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])

if len(image_files) != len(json_files):
    print("Warning: Number of images and JSON files do not match!")

for img_file, json_file in zip(image_files, json_files):
    img_path = os.path.join(image_dir, img_file)
    json_path = os.path.join(json_dir, json_file)
    output_file = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}.txt")  # 输出的txt文件

    image = cv2.imread(img_path)
    if image is None:
        print(f"Error reading image: {img_path}")
        continue

    with open(json_path, 'r') as f:
        data = json.load(f)

    with open(output_file, 'w') as f:
        for obj in data['objects']:
            x, y, w, h = obj['bndbox']

            for contour in obj['contours']:
                contour = np.array(contour).squeeze()
                contour[:, 0] += x  
                contour[:, 1] += y  

                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)  
                box = np.int0(box)  

                cv2.polylines(image, [box], isClosed=True, color=(0, 0, 255), thickness=2)

                center = rect[0]
                cv2.circle(image, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)

                annotation = f"{box[0][0]} {box[0][1]} {box[1][0]} {box[1][1]} {box[2][0]} {box[2][1]} {box[3][0]} {box[3][1]} plane 0\n"
                f.write(annotation)

    print(f"Annotations saved to {output_file}")

    cv2.imshow(f"Image with Annotations - {img_file}", image)
    key = cv2.waitKey(0)
    if key == 27:  # 按下Esc键退出
        break

cv2.destroyAllWindows()
