import sys
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

model_path = "runs\\detect\\train\\weights\\best.pt"

try:
    print(f"Loading YOLO model from {model_path}")
    model = YOLO(model_path)
    class_names = model.names
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    sys.exit(f"Failed to load YOLO model: {e}")

image_path = 'test.png'

img = cv2.imread(image_path)

if img is None:
    print("Error loading image")
    sys.exit()

results = model(img)

for result in results:
    boxes = result.boxes 

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        confidence = box.conf[0].cpu().numpy()
        confidence = float(confidence)

        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = class_names[cls_id]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        image_height, image_width = img.shape[:2]
        start_point = (image_width // 2, image_height)
        end_point = (center_x, center_y)

        label = f"{cls_name}: {confidence:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_origin = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)
        cv2.rectangle(img, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)

output_path = 'test_detected.png'
cv2.imwrite(output_path, img)
print(f"Detection result saved to {output_path}")

img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_display)
plt.axis('off')
plt.show()
