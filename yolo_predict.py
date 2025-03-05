from ultralytics import YOLO
from PIL import Image

# Load a model
model = YOLO("./best.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(["rgb_image_00000.png", "rgb_image_00129.png", "rgb_image_00545.png"], conf = 0.5, iou = 0.4)  # return a list of Results objects

# Process results list
for idx, result in enumerate(results):
    print(result)
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    im_bgr = result.plot(labels=False)  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
    im_rgb.save("result{}.jpg".format(idx))

    # result.save(filename="result{}.jpg".format(idx))  # save to disk