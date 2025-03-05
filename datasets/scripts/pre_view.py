import cv2
import os
import numpy as np

image_dir = "./data_white/images/train"  
label_dir = "./data_white/labels/train_original" 

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

if len(image_files) != len(label_files):
    print("Warning: Number of images and labels do not match!")

for img_file, lbl_file in zip(image_files, label_files):
    img_path = os.path.join(image_dir, img_file)
    lbl_path = os.path.join(label_dir, lbl_file)
    
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error reading image: {img_path}")
        continue

    try:
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    print(f"Invalid format in {lbl_path}: {line}")
                    continue

                
                x1, y1, x2, y2, x3, y3, x4, y4 = map(int, parts[:8])
                label = parts[8] 

                points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

                # text_position = (x1, y1 - 10)
                # cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error reading label file: {lbl_path}, Error: {e}")
        continue

    cv2.imshow("Annotation Viewer", image)
    print(f"Displaying {img_file} with annotations from {lbl_file}")

    key = cv2.waitKey(0)
    if key == 27:  
        break

cv2.destroyAllWindows()
