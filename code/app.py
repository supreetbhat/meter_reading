import cv2
import pytesseract
import json
from ultralytics import YOLO
from PIL import Image

# Load trained YOLO model
model = YOLO("/Users/supreetbhat/Github/meter_reading/runs/detect/train/weights/best.pt")

# Run detection
results = model.predict(source="/Users/supreetbhat/Github/meter_reading/datasets/meter-reader-detection/test/images/571_Genus_1PH_4_jpg.rf.dca195db211296bae948b7667b42c231.jpg", save=False)

# Initialize JSON with default 0 values
output = {"meter_readings": [{"value": 0, "confidence": 0}],
          "serial_number": [{"value": 0, "confidence": 0}]}

# Loop through detections
for r in results:
    img = cv2.imread("/Users/supreetbhat/Github/meter_reading/datasets/meter-reader-detection/test/images/571_Genus_1PH_4_jpg.rf.dca195db211296bae948b7667b42c231.jpg")
    for box in r.boxes:
        cls = int(box.cls[0])  # class id
        conf = float(box.conf[0])  # confidence
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # coordinates

        # Crop detected region
        crop = img[y1:y2, x1:x2]

        # Convert to PIL for OCR
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        # Run OCR
        text = pytesseract.image_to_string(pil_crop, config="--psm 6").strip()

        # Replace 0 with detected value
        if cls == 0:  # meter_readings
            output["meter_readings"] = [{"value": text, "confidence": conf}]
        elif cls == 1:  # serial_number
            output["serial_number"] = [{"value": text, "confidence": conf}]

# Print JSON
print(json.dumps(output, indent=4))
