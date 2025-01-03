import logging
import cv2
import torch
import time
import statistics
import warnings
from datetime import datetime
from configParams import Parameters
from database.db_entries_utils import db_entries_time  # Import the function for database logging
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('torchvision').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
params = Parameters()

# Device setup
def get_device():
    if torch.cuda.is_available():
        torch.device("cuda")
        return 0
    elif cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        return torch.device("cpu")  # PyTorch doesn't support OpenCL, so set it for OpenCV
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# Load models for plate and character detection
model_plate = torch.hub.load('yolov5', 'custom', params.modelPlate_path, source='local', device=device, force_reload=True)
model_char = torch.hub.load('yolov5', 'custom', params.modelCharX_path, source='local', device=device, force_reload=True)

# RTSP or video source setup
source = params.rtps if params.rtps else 0  # If params.rtps is defined, use it; otherwise, default to the webcam.
cap = cv2.VideoCapture(source)

# Function to detect characters on the license plate
def detect_plate_chars(cropped_plate):
    chars, confidences, char_detected = [], [], []
    results = model_char(cropped_plate)
    detections = sorted(results.pred[0], key=lambda x: x[0])  # Sort by x-coordinate
    for det in detections:
        conf = det[4]
        if conf > 0.5:
            cls = int(det[5].item())
            char = params.char_id_dict.get(str(cls), '')
            chars.append(char)
            confidences.append(conf.item())
            char_detected.append(det.tolist())
    char_conf_avg = round(statistics.mean(confidences) * 100) if confidences else 0
    return ''.join(chars), char_conf_avg

# Function to draw a rectangle around the detected plate
def highlight_plate(image, x_min, y_min, x_max, y_max):
    cv2.rectangle(image, (x_min - 3, y_min - 3), (x_max + 3, y_max + 3), color=(0, 0, 255), thickness=3)

# Main loop for processing frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not retrieve frame.")
        time.sleep(1)  # Retry delay
        continue

    # License plate detection
    plate_results = model_plate(frame).pandas().xyxy[0]
    for _, plate in plate_results.iterrows():
        plate_conf = int(plate['confidence'] * 100)
        if plate_conf >= 85:
            x_min, y_min, x_max, y_max = int(plate['xmin']), int(plate['ymin']), int(plate['xmax']), int(plate['ymax'])
            # highlight_plate(frame, x_min, y_min, x_max, y_max)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cropped_plate = frame[y_min:y_max, x_min:x_max]
            plate_text, char_conf_avg = detect_plate_chars(cropped_plate)

            # Show plate text on the video frame
            cv2.putText(frame, f"Plate: {plate_text}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # Call db_entries_time function to handle screenshot saving and any other logic
            if char_conf_avg >= 75 and len(plate_text) >= 8:
                db_entries_time(
                    number=plate_text,
                    charConfAvg=char_conf_avg,
                    plateConfAvg=plate_conf,
                    croppedPlate=cropped_plate,
                    status="Active",
                    frame=frame  # Pass the full frame for screenshot saving
                )

    # Calculate and display FPS on the frame
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame without altering colors
    cv2.imshow('License Plate Detection', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
