# Automated Electric Meter Reader

A project that uses a custom-trained **YOLOv8** model to detect meter readings and serial numbers on electric meters, and **Tesseract OCR** to extract the text from these regions.

This system automates the manual and error-prone process of meter reading by providing a fast and accurate pipeline that takes an image as input and returns a structured JSON object with the extracted data.

---

## Features

- **Object Detection:** Uses a custom-trained YOLOv8 model to accurately locate:
    
    - Meter Readings (Class 0)
        
    - Serial Numbers (Class 1)
        
- **Optical Character Recognition (OCR):** Employs Tesseract to extract text from the detected regions.
    
- **Structured Output:** Provides results in a simple, easy-to-parse JSON format, including confidence scores for each detection.
    
- **Modular:** Easy to swap in new models or change OCR configurations.
    

---

## How it Works

The project follows a two-stage pipeline:

1. **Detection:** The system loads the custom-trained YOLOv8 model (`best.pt`). An input image is passed to the model, which returns a list of bounding boxes for all detected objects (meter readings and serial numbers).
    
2. **Recognition:** For each detected bounding box:
    
    - The region is **cropped** from the original image.
        
    - The crop is converted from an OpenCV format to a PIL (Pillow) image.
        
    - **Tesseract OCR** is run on the PIL image to extract the text. We use `config="--psm 6"` which assumes the crop is a single uniform block of text.
        
    - The extracted text and the model's detection confidence score are stored.
        
3. **Output:** The final data is compiled and printed as a JSON object.
    

---

## 📦 Project Structure

```
your-project-directory/
│
├── runs/detect/train/weights/
│   └── best.pt           # Your trained YOLOv8 model
│
├── datasets/
│   └── ...               # Your training and test images
│
├── read_meter.py         # The main Python script (your code)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🛠️ Requirements & Installation

### 1. Python Dependencies

This project requires the following Python libraries. You can install them all using `pip`.

```
ultralytics
opencv-python
pytesseract
pillow
```

Create a `requirements.txt` file with the contents above and run:

Bash

```
pip install -r requirements.txt
```

### 2. Tesseract-OCR Engine

`pytesseract` is just a Python wrapper. You must install the Tesseract-OCR engine itself on your system.

**On macOS:**

Bash

```
brew install tesseract
```

**On Ubuntu/Debian:**

Bash

```
sudo apt update
sudo apt install tesseract-ocr
```

**On Windows:** Download and run the official installer from the [Tesseract at UB-Mannheim](https://www.google.com/search?q=https://github.com/UB-Mannheim/tesseract/wiki) repository. Make sure to add the Tesseract installation directory to your system's `PATH`.

---

## 🚀 How to Run

The provided script (`read_meter.py`) is set up to run inference on a single, hardcoded image.

1. **Ensure paths are correct:**
    
    - Make sure the path to your YOLO model (`best.pt`) is correct.
        
    - Make sure the `source` image path for `model.predict()` and the `cv2.imread()` path are correct.
        
2. **Execute the script:**
    
    Bash
    
    ```
    python read_meter.py
    ```
    

### Example Output

Running the script will process the image and print a JSON object to the console, similar to this:

JSON

```
{
    "meter_readings": [
        {
            "value": "123456",
            "confidence": 0.9458023309707642
        }
    ],
    "serial_number": [
        {
            "value": "SN987654",
            "confidence": 0.8912345170974731
        }
    ]
}
```

---

## 🏋️ Model Training

The model used (`best.pt`) was custom-trained using YOLOv8 on a labeled dataset of electric meter images.

- **Dataset:** The dataset was annotated with bounding boxes for two classes:
    
    - `0`: `meter_reading`
        
    - `1`: `serial_number`
        
- **Training:** The model was trained using the `ultralytics` library. For information on how to train your own model, please refer to the [official YOLOv8 documentation](https://docs.ultralytics.com/tasks/detect/).
    

---

## 💡 Future Improvements

- **CLI Arguments:** Modify the script to accept the image path as a command-line argument instead of hardcoding it.
    
- **Image Pre-processing:** Add image pre-processing steps (e.g., grayscale, thresholding, denoising) to the cropped regions _before_ sending them to Tesseract to improve OCR accuracy.
    
- **Web API:** Wrap the entire pipeline in a simple web API (using Flask or FastAPI) to allow for easy integration with other services.
    
- **Batch Processing:** Add functionality to process an entire directory of images instead of just one.
