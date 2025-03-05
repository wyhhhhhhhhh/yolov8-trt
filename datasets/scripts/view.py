import cv2
import os
import numpy as np

image_dir = "./data_white/images/train"  
label_dir = "./data_white/labels/train" 

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

# 可视化函数
def visualize_labels(image_path, label_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("图像加载失败，请检查路径！")
        return
    image_h, image_w, _ = image.shape

    with open(label_path, "r") as file:
        labels = file.readlines()

    for label in labels:
        parts = label.strip().split()
        class_index = int(parts[0])
        coords = list(map(float, parts[1:]))

        points = [(int(x * image_w), int(y * image_h)) for x, y in zip(coords[::2], coords[1::2])]
        points = points + [points[0]]  

        cv2.polylines(image, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

        # cv2.putText(image, f"Class {class_index}", points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


    cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        

for img_file, lbl_file in zip(image_files, label_files):
    image_path = os.path.join(image_dir, img_file)
    label_path = os.path.join(label_dir, lbl_file)
    
    print(f"Displaying {img_file} with annotations from {lbl_file}")
    visualize_labels(image_path, label_path)

    key = cv2.waitKey(0)
    if key == 27:  
        break

cv2.destroyAllWindows()
