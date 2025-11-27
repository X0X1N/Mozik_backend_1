import cv2
from ultralytics import YOLO
import numpy as np

# Global variable to store the loaded YOLO model
_yolo_model = None
_yolo_model_path = r"C:\yolov5\runs\train\person_finetune_v24\weights\best.pt"

def load_yolo_model():
    """
    Loads the YOLOv5 model from the specified path.
    """
    global _yolo_model
    if _yolo_model is None:
        print(f"Loading YOLO model from {_yolo_model_path}...")
        _yolo_model = YOLO(_yolo_model_path)
        print("YOLO model loaded.")
    return _yolo_model

def detect_faces_yolo(image: np.ndarray) -> list:
    """
    Detects faces in the given image using the loaded YOLO model.

    Args:
        image (np.ndarray): The input image (BGR format).

    Returns:
        list: A list of detected bounding boxes in (x, y, w, h) format.
    """
    model = load_yolo_model()
    
    # YOLO expects RGB images, but OpenCV reads BGR. Convert if necessary.
    # Assuming the input 'image' is BGR from OpenCV
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(rgb_image, verbose=False)  # verbose=False to suppress output

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # box.xyxy returns [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w = x2 - x1
            h = y2 - y1
            detections.append((x1, y1, w, h))
    return detections

if __name__ == '__main__':
    # Example usage:
    # Create a dummy image for testing
    dummy_image = np.zeros((640, 480, 3), dtype=np.uint8)
    # Draw a rectangle to simulate a face
    cv2.rectangle(dummy_image, (100, 100), (200, 200), (0, 255, 0), 2)

    print("Testing YOLO face detection...")
    detected_boxes = detect_faces_yolo(dummy_image)
    print(f"Detected boxes: {detected_boxes}")

    # You would typically load a real image here
    # img_path = "path/to/your/image.jpg"
    # img = cv2.imread(img_path)
    # if img is not None:
    #     detected_boxes_real = detect_faces_yolo(img)
    #     print(f"Detected boxes in real image: {detected_boxes_real}")
    # else:
    #     print(f"Could not load image from {img_path}")

