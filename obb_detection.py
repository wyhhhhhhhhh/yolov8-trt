import json
import sys
import os

import cv2
import numpy as np

from ultralytics import YOLO
from PIL import Image

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

    global model_file, model
    model_file = os.path.join(data_folder, config['model_files']['obb_model_file'])
    model = YOLO(model_file)

    global global_counter, cur_data_folder
    use_counter = config['use_counter']
    if use_counter:
        cur_data_folder = os.path.join(data_folder, str(global_counter))
        if not os.path.exists(cur_data_folder):
            os.makedirs(cur_data_folder, exist_ok=True)
        global_counter = global_counter + 1

# Load a model
model_file = os.path.join(data_folder, config['model_files']['obb_model_file'])
model = YOLO(model_file)

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
    """
    Converts an oriented bounding box (OBB) with 4 points to a binary mask.

    Parameters:
        obb_points (list or np.array): A list or array of 8 elements representing the 4 points (x1, y1, x2, y2, x3, y3, x4, y4).
        mask_size (tuple): A tuple specifying the size of the mask (height, width).

    Returns:
        np.array: A binary mask with the OBB filled with 255 (white).
    """
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
    """
    Converts a list of OBBs into a list of binary masks.

    Parameters:
        obbs (np.array or list): A 2D array or list of shape (n, 8), where n is the number of OBBs, 
                                 and each row represents 8 elements (x1, y1, x2, y2, x3, y3, x4, y4).
        mask_size (tuple): A tuple specifying the size of the mask (height, width).

    Returns:
        list: A list of binary masks corresponding to each OBB, each mask is an ndarray of shape `mask_size`.
    """
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


def detect(left_image, right_image):
    update_config()
    left_image_pil = Image.fromarray(left_image).convert('RGB')
    right_image_pil = Image.fromarray(right_image).convert('RGB')
    left_image = np.array(left_image_pil)
    right_image = np.array(right_image_pil)

    # crop images
    cropped_left_img, left_start_coord = crop_image(left_image, is_left=True)
    cropped_right_img, right_start_coord = crop_image(right_image, is_left=False)
    cropped_left_img = cv2.imread("/mnt/c/Users/mech-mind/Desktop/zhanhui/blenderproc/examples/basics/stereo_camera/vision_steps/stereo_ai/source/rgb_image_lcc.png")
    cropped_right_img = cv2.imread("/mnt/c/Users/mech-mind/Desktop/zhanhui/blenderproc/examples/basics/stereo_camera/vision_steps/stereo_ai/source/rgb_image_rcc.png")


    # Run batched inference on a list of images
    # results = model([left_image, right_image], conf=confidence, iou=nms_iou, imgsz=int(left_image.shape[1]/scale_down))  # return a list of Results objects
    # results = model([left_image, right_image], conf=confidence, iou=nms_iou)  # return a list of Results objects
    results = model([cropped_left_img, cropped_right_img], conf=confidence, iou=nms_iou)  # return a list of Results objects
    masks_list = []
    downsampled_masks_list = []
    original_images = [left_image_pil, right_image_pil]
    original_left_tops = [left_start_coord, right_start_coord]
    # Process results list
    for idx, result in enumerate(results):
        if save_data: 
            save_image_result_with_cropped_image(result, original_images[idx], original_left_tops[idx], idx, cur_data_folder)
            # save_image_result(result, idx, cur_data_folder)
        obbs = result.obb.cpu().numpy().xyxyxyxy  # Oriented boxes object for OBB outputs
        obbs = obbs + original_left_tops[idx]
        downsampled_obbs = obbs / scale_down
        cur_masks = generate_masks_for_multiple_obbs(obbs, mask_size=(left_image.shape[0], left_image.shape[1]))
        downsampled_masks = generate_masks_for_multiple_obbs(downsampled_obbs, mask_size=(int(left_image.shape[0]/scale_down), int(left_image.shape[1]/scale_down)))
        masks_list.append(cur_masks)
        downsampled_masks_list.append(downsampled_masks)
    
    return downsampled_masks_list[0], downsampled_masks_list[1], masks_list[0], masks_list[1]



if __name__ == "__main__":
    # left_image = Image.open("rgb_image_00000L.png").convert("RGB")
    # right_image = Image.open("rgb_image_00000R.png").convert("RGB")
    left_image = Image.open("rgb_image_l.png").convert("RGB")
    right_image = Image.open("rgb_image_r.png").convert("RGB")
    # left_image = Image.open("/mnt/c/Users/mech-mind/Desktop/zhanhui/blenderproc/examples/basics/stereo_camera/vision_steps/stereo_ai/source/rgb_image_l.png").convert("RGB")
    # right_image = Image.open("/mnt/c/Users/mech-mind/Desktop/zhanhui/blenderproc/examples/basics/stereo_camera/vision_steps/stereo_ai/source/rgb_image_r.png").convert("RGB")
    left_image = np.array(left_image)
    right_image = np.array(right_image)
    detect(left_image, right_image)