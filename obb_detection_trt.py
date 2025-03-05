import json
import sys
import os

import cv2
import numpy as np

from ultralytics import YOLO
from PIL import Image

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import torch

# Get the current script's directory
current_script_directory = os.path.dirname(os.path.abspath(__file__))
# Add the current script's directory to the Python path
sys.path.append(current_script_directory)
print(current_script_directory)

config_file = os.path.join(current_script_directory, "config.json")
with open(config_file, 'r') as file:
    config = json.load(file)

current_data_folder = config['data_cache']
data_folder = os.path.join(current_script_directory, current_data_folder)

use_counter = config['use_counter']
global_counter = 0
cur_data_folder = os.path.join(data_folder, str(global_counter))
if not os.path.exists(cur_data_folder):
    os.makedirs(cur_data_folder, exist_ok=True)

confidence = config['obb_confidence']
nms_iou = config['obb_nms_iou']
scale_down = config['scale_down']
reload_config = config["update_config"]
save_data = config["save_data"]
detection_size = tuple(config['detection_size'])
target_size = tuple(config['target_size'])
left_roi = config['left_roi']
right_roi = config['right_roi']

def compute_adjusted_roi(union_top_left, union_bottom_right, 
                         new_width, new_height, image_width, image_height):
    """
    Compute the union of two ROIs, expand it into a new size based on the center, 
    and adjust the ROI to fit within the image bounds.
    
    Args:
    - roi1_top_left (tuple): (x1, y1) coordinates of the top-left corner of the first ROI.
    - roi1_bottom_right (tuple): (x2, y2) coordinates of the bottom-right corner of the first ROI.
    - roi2_top_left (tuple): (x1, y1) coordinates of the top-left corner of the second ROI.
    - roi2_bottom_right (tuple): (x2, y2) coordinates of the bottom-right corner of the second ROI.
    - new_width (int): The desired width of the expanded ROI.
    - new_height (int): The desired height of the expanded ROI.
    - image_width (int): The width of the image (to check for out-of-bounds).
    - image_height (int): The height of the image (to check for out-of-bounds).
    
    Returns:
    - tuple: (adjusted_top_left, adjusted_bottom_right)
        - adjusted_top_left (tuple): (x_min, y_min) coordinates of the top-left corner of the adjusted ROI.
        - adjusted_bottom_right (tuple): (x_max, y_max) coordinates of the bottom-right corner of the adjusted ROI.
        - hint (str): A message indicating whether the ROI was adjusted.
    """
    # Step 2: Find the center of the union ROI
    center_x = (union_top_left[0] + union_bottom_right[0]) // 2
    center_y = (union_top_left[1] + union_bottom_right[1]) // 2

    # Step 3: Calculate half-width and half-height for the new ROI
    half_width = new_width // 2
    half_height = new_height // 2

    # Step 4: Compute the expanded top-left and bottom-right coordinates
    expanded_top_left = (center_x - half_width, center_y - half_height)
    expanded_bottom_right = (center_x + half_width, center_y + half_height)

    if (expanded_top_left[0] > union_top_left[0]) or (expanded_top_left[1] > union_top_left[1]) or (expanded_bottom_right[0] < union_top_left[0]) or (expanded_bottom_right[1] < expanded_bottom_right[1]):
        raise ValueError(f"the cropped size cannot cover the stereo roi.\n-cropped roi=({expanded_top_left},{expanded_bottom_right})\n-stereo roi=({union_top_left},{union_bottom_right})")

    # Step 5: Adjust if out of bounds
    # Initialize hint message
    hint = "ROI is within bounds."

    # Check if the top-left goes out of bounds
    if expanded_top_left[0] < 0:
        expanded_bottom_right = (expanded_bottom_right[0] + (0 - expanded_top_left[0]), expanded_bottom_right[1])
        expanded_top_left = (0, expanded_top_left[1])
        hint = "top_left_x of ROI adjusted to fit within bounds."
        print(hint)
        
    if expanded_top_left[1] < 0:
        expanded_bottom_right = (expanded_bottom_right[0], expanded_bottom_right[1] + (0 - expanded_top_left[1]))
        expanded_top_left = (expanded_top_left[0], 0)
        hint = "top_left_y of ROI adjusted to fit within bounds."
        print(hint)
    
    # Check if the bottom-right goes out of bounds
    if expanded_bottom_right[0] > image_width:
        expanded_top_left = (expanded_top_left[0] - (expanded_bottom_right[0] - image_width), expanded_top_left[1])
        expanded_bottom_right = (image_width, expanded_bottom_right[1])
        hint = "bottom_right_x of ROI adjusted to fit within bounds."
        print(hint)
    
    if expanded_bottom_right[1] > image_height:
        expanded_top_left = (expanded_top_left[0], expanded_top_left[1] - (expanded_bottom_right[1] - image_height))
        expanded_bottom_right = (expanded_bottom_right[0], image_height)
        hint = "bottom_right_y of ROI adjusted to fit within bounds."
        print(hint)

    print(hint)

    return expanded_top_left, expanded_bottom_right

expand_left_roi = compute_adjusted_roi(left_roi[0], left_roi[1], detection_size[0], detection_size[1], target_size[0], target_size[1])
expand_right_roi = compute_adjusted_roi(right_roi[0], right_roi[1], detection_size[0], detection_size[1], target_size[0], target_size[1])

def update_config():
    global config
    with open(config_file, 'r') as file:
        config = json.load(file)
    global reload_config
    reload_config = config["update_config"]
    if not reload_config:
        return

    global confidence 
    global nms_iou
    confidence = config['obb_confidence']
    nms_iou = config['obb_nms_iou']
    global scale_down
    scale_down = config['scale_down']

    global save_data
    save_data = config['save_data']

    global detection_size, target_size
    detection_size = tuple(config['detection_size'])
    target_size = tuple(config['target_size'])

    global left_roi, right_roi
    left_roi = config['left_roi']
    right_roi = config['right_roi']

    global expand_left_roi, expand_right_roi
    expand_left_roi = compute_adjusted_roi(left_roi[0], left_roi[1], detection_size[0], detection_size[1], target_size[0], target_size[1])
    expand_right_roi = compute_adjusted_roi(right_roi[0], right_roi[1], detection_size[0], detection_size[1], target_size[0], target_size[1])

    global global_counter, cur_data_folder
    use_counter = config['use_counter']
    if use_counter:
        cur_data_folder = os.path.join(data_folder, str(global_counter))
        if not os.path.exists(cur_data_folder):
            os.makedirs(cur_data_folder, exist_ok=True)
        global_counter = global_counter + 1

# Load a model
model_file = os.path.join(data_folder, config['model_files']['obb_model_trt'])
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(trt_runtime, engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    return trt_runtime.deserialize_cuda_engine(engine_data)

trt_runtime = trt.Runtime(TRT_LOGGER)
engine = load_engine(trt_runtime, model_file)
if engine is None:
    raise ValueError("Failed to load TensorRT engine!")
context = engine.create_execution_context()

def save_image_result(result, idx, cur_data_folder):
    im_bgr = result.plot(labels=False)  # BGR-order numpy array
    # im_bgr = result.plot(font_size=1, conf=True)  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
    im_file = os.path.join(cur_data_folder, "result{}.jpg".format(idx))
    im_rgb.save(im_file)

def save_image_result_with_cropped_image(result, original_image, start_coord, idx, cur_data_folder):
    im_bgr = result.plot(labels=False)  # BGR-order numpy array
    # im_bgr = result.plot(font_size=1, conf=True)  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
    original_image.paste(im_rgb, start_coord)
    im_file = os.path.join(cur_data_folder, "detection_result{}.jpg".format(idx))
    original_image.save(im_file)

def obb_to_mask(obb_points, mask_size=(256, 256)):
    # Convert the OBB points to a 4x2 numpy array (each point is (x, y))
    obb_points = np.array(obb_points).reshape((4, 2))

    # Create a blank mask (initialize to zeros)
    mask = np.zeros(mask_size, dtype=np.uint8)

    # Convert OBB points to integer coordinates
    obb_points = np.int32(obb_points)

    # Fill the polygon (OBB) on the mask
    cv2.fillPoly(mask, [obb_points], 255)  # Fill the polygon with 255 (white)

    return mask

def generate_masks_for_multiple_obbs(obbs, mask_size=(256, 256)):
    masks = []
    
    # Loop through each OBB and convert it to a mask
    for obb in obbs:
        mask = obb_to_mask(obb, mask_size)
        masks.append(mask)
    
    return masks

def crop_image(image, is_left=True):
    if is_left:
        left_top, right_bottom = expand_left_roi
    else:
        left_top, right_bottom = expand_right_roi
    cropped_image = image[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
    return cropped_image, left_top

def allocate_buffers(engine):
    input_shape = engine.get_tensor_shape(engine.get_tensor_name(0))
    output_shape = engine.get_tensor_shape(engine.get_tensor_name(1))
    
    input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize
    output_size = trt.volume(output_shape) * np.dtype(np.float32).itemsize
    input_buffer_gpu = cuda.mem_alloc(input_size)
    output_buffer_gpu = cuda.mem_alloc(output_size)
    
    return input_buffer_gpu, output_buffer_gpu

def infer_tensorrt(context, input_tensor, input_buffer_gpu, output_buffer_gpu):
    cuda.memcpy_htod(input_buffer_gpu, input_tensor)
    
    bindings = [int(input_buffer_gpu), int(output_buffer_gpu)]
    success = context.execute_v2(bindings=bindings)
    if not success:
        raise RuntimeError("TensorRT推理失败！")
    
    output_buffer = np.empty(context.get_tensor_shape(engine.get_tensor_name(1)), dtype=np.float32)
    cuda.memcpy_dtoh(output_buffer, output_buffer_gpu)
    
    return output_buffer

def preprocess(input_image):
    def letterbox_pad(image, target_size=(640, 640), color=(114, 114, 114)):
        h, w = image.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)  # 计算缩放比例
        nw, nh = int(w * scale), int(h * scale)  # 计算缩放后的尺寸

        image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)  # 等比例缩放

        pad_w = target_size[0] - nw
        pad_h = target_size[1] - nh
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)

        padded_image = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return padded_image, (nw, nh, left, top)  # 返回填充后的图像和 padding 信息
    
    input_image, pad_info = letterbox_pad(input_image, (640, 640)) 
    cv2.imwrite("dwqd.png", input_image)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  
    input_image = np.transpose(input_image, (2, 0, 1))  # HWC → CHW
    input_image = input_image.astype(np.float32) / 255.0  # 归一化
    input_tensor = np.expand_dims(input_image, axis=0)  # (1, 3, 640, 640)
    
    return np.ascontiguousarray(input_tensor, dtype=np.float32), pad_info

def process_obb_output(obb_output, confidence_threshold=0.5, nms_iou_threshold=0.4):
    obb_output = obb_output.squeeze(0) 
    cx, cy, w, h = obb_output[0], obb_output[1], obb_output[2], obb_output[3]  
    class_probs = obb_output[4:19]  
    angle = obb_output[19]  
    conf = np.max(class_probs, axis=0)  
    class_id = np.argmax(class_probs, axis=0)  # 置信度最高的类别索引

    mask = conf > confidence_threshold
    selected_boxes = np.stack([cx[mask], cy[mask], w[mask], h[mask], angle[mask]], axis=-1)
    selected_scores = conf[mask]
    selected_classes = class_id[mask]

    final_obbs = np.column_stack([selected_boxes, selected_scores, selected_classes])  # (N, 7)

    return final_obbs

def nms_obb(obbs, iou_threshold=0.4):
    def obb_iou(box1, box2):
        rect1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        rect2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

        poly1 = cv2.boxPoints(rect1)
        poly2 = cv2.boxPoints(rect2)

        poly1 = np.intp(poly1)
        poly2 = np.intp(poly2)

        # 计算交集
        inter_area = cv2.intersectConvexConvex(poly1, poly2)[0]
        if inter_area is None:
            inter_area = 0

        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]

        iou = inter_area / (area1 + area2 - inter_area)
        return iou

    if len(obbs) == 0:
        return []

    obbs = obbs[obbs[:, 5].argsort()[::-1]]

    selected_obbs = []
    while len(obbs) > 0:
        best_box = obbs[0]
        selected_obbs.append(best_box)

        obbs = obbs[1:]
        ious = np.array([obb_iou(best_box, box) for box in obbs])
        obbs = obbs[ious < iou_threshold]

    return np.array(selected_obbs)

def convert_obb_to_corners(select_obb_l):
    def get_obb_corners(cx, cy, w, h, angle):
        corners = np.array([
            [w/2,  h/2],   # 右上
            [-w/2, h/2],   # 左上
            [-w/2, -h/2],  # 左下
            [w/2,  -h/2]   # 右下
        ], dtype=np.float32)
        
        # 旋转矩阵
        rot_mat = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ], dtype=np.float32)
        
        # 应用旋转
        rotated = np.dot(corners, rot_mat.T)
        # 平移至中心点坐标
        rotated[:, 0] += cx
        rotated[:, 1] += cy
        return rotated

    select_obb = np.asarray(select_obb_l, dtype=np.float32)
    obb_corners = np.empty((len(select_obb), 4, 2), dtype=np.float32)
    
    for i, obb in enumerate(select_obb):
        cx, cy, w, h, angle, _, _ = obb
        obb_corners[i] = get_obb_corners(cx, cy, w, h, angle)
        
    return obb_corners

def draw_scaled_obb(image, obbs):
    for obb in obbs:
        cx, cy, w, h, angle, score, class_id = obb
        rect = ((cx, cy), (w, h), np.float32(angle * 180 / np.pi))  # 角度转换为度
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
        label = f"Class {int(class_id)}: {score:.2f}"
        cv2.putText(image, label, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

def scale_obb_to_original(obb_results, original_size, pad_info):
    """将推理得到的 OBB 框从 (640, 640) 还原到 (640, 480)"""
    nw, nh, left, top = pad_info  # 获取 padding 信息
    scale_x = original_size[0] / nw  
    scale_y = original_size[1] / nh  

    scaled_obbs = obb_results.copy()
    scaled_obbs[:, 0] = (obb_results[:, 0] - left) * scale_x  # 缩放中心点 cx
    scaled_obbs[:, 1] = (obb_results[:, 1] - top) * scale_y  # 缩放中心点 cy
    scaled_obbs[:, 2] *= scale_x  # 缩放 w
    scaled_obbs[:, 3] *= scale_y  # 缩放 h

    return scaled_obbs


input_buffer_gpu, output_buffer_gpu = allocate_buffers(engine)
def detect(left_image, right_image):
    update_config()
    left_image_pil = Image.fromarray(left_image).convert('RGB')
    right_image_pil = Image.fromarray(right_image).convert('RGB')
    
    left_image = np.array(left_image_pil)
    right_image = np.array(right_image_pil)

    cropped_left_img, left_start_coord = crop_image(left_image, is_left=True)
    cropped_right_img, right_start_coord = crop_image(right_image, is_left=False)
    # cv2.imwrite("111.png", cropped_left_img)
    cropped_left_img = cv2.imread("/mnt/c/Users/mech-mind/Desktop/zhanhui/blenderproc/examples/basics/stereo_camera/vision_steps/stereo_ai/source/rgb_image_lcc.png")
    cropped_right_img = cv2.imread("/mnt/c/Users/mech-mind/Desktop/zhanhui/blenderproc/examples/basics/stereo_camera/vision_steps/stereo_ai/source/rgb_image_rcc.png")
 
    left_tensor, pad_left = preprocess(cropped_left_img)
    right_tensor, pad_right = preprocess(cropped_right_img)

    left_result = infer_tensorrt(context, left_tensor, input_buffer_gpu, output_buffer_gpu)
    right_result = infer_tensorrt(context, right_tensor, input_buffer_gpu, output_buffer_gpu)

    select_obb_l = nms_obb(process_obb_output(left_result, 0.8, 0.5))
    select_obb_r = nms_obb(process_obb_output(right_result, 0.8, 0.5))
    scaled_obb_l = scale_obb_to_original(select_obb_l, original_size=(640, 480), pad_info=pad_left)
    scaled_obb_r = scale_obb_to_original(select_obb_r, original_size=(640, 480), pad_info=pad_right)

    image_with_obbs_rescaled_l = draw_scaled_obb(cropped_left_img, scaled_obb_l)
    image_with_obbs_rescaled_r = draw_scaled_obb(cropped_right_img, scaled_obb_r)
    cv2.imwrite("1.png", image_with_obbs_rescaled_l)
    cv2.imwrite("2.png", image_with_obbs_rescaled_r)

    obb_corners_l = convert_obb_to_corners(select_obb_l)
    obb_corners_r = convert_obb_to_corners(select_obb_r)

    cur_masks_l = generate_masks_for_multiple_obbs(obb_corners_l*scale_down, mask_size=(left_image.shape[0], left_image.shape[1]))
    cur_masks_r = generate_masks_for_multiple_obbs(obb_corners_r*scale_down, mask_size=(left_image.shape[0], left_image.shape[1]))
    downsampled_masks_l = generate_masks_for_multiple_obbs(obb_corners_l, mask_size=(int(left_image.shape[0]/scale_down), int(left_image.shape[1]/scale_down)))
    downsampled_masks_r = generate_masks_for_multiple_obbs(obb_corners_r, mask_size=(int(left_image.shape[0]/scale_down), int(left_image.shape[1]/scale_down)))
    
    return downsampled_masks_l, downsampled_masks_r, cur_masks_l, cur_masks_r



if __name__ == "__main__":
    # left_image = Image.open("rgb_image_l.png").convert("RGB")
    # right_image = Image.open("rgb_image_r.png").convert("RGB")
    left_image = Image.open("/mnt/c/Users/mech-mind/Desktop/zhanhui/blenderproc/examples/basics/stereo_camera/vision_steps/stereo_ai/source/rgb_image_l.png").convert("RGB")
    right_image = Image.open("/mnt/c/Users/mech-mind/Desktop/zhanhui/blenderproc/examples/basics/stereo_camera/vision_steps/stereo_ai/source/rgb_image_r.png").convert("RGB")
    left_image = np.array(left_image)
    right_image = np.array(right_image)
    detect(left_image, right_image)