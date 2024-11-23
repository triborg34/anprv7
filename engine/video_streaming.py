import logging
import cv2
import torch
import time
import statistics
import warnings
import asyncio
import base64
from collections import deque
import websockets
from datetime import datetime
from configParams import Parameters
from database.db_entries_utils import db_entries_time  # Import the function for database logging

logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('torchvision').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
params = Parameters()

# WebSocket parameters
port = 5000
host = '127.0.0.1'

# Frame buffer for asynchronous handling
frame_buffer = deque(maxlen=100)  # Buffer to store processed frames

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

# Video source setup
cap = cv2.VideoCapture(params.video_path)

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

async def capture_and_process_frames():
    """Capture frames from the video file, process them, and store in the buffer."""
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error in reading the frame.")
            break

        # License plate detection
        plate_results = model_plate(frame).pandas().xyxy[0]
        for _, plate in plate_results.iterrows():
            plate_conf = int(plate['confidence'] * 100)
            if plate_conf >= 85:
                x_min, y_min, x_max, y_max = int(plate['xmin']), int(plate['ymin']), int(plate['xmax']), int(plate['ymax'])
                highlight_plate(frame, x_min, y_min, x_max, y_max)
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

        # Encode the processed frame to JPEG
        _, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        data = base64.b64encode(encoded).decode('utf-8')

        # Add the frame to the buffer
        if len(frame_buffer) < frame_buffer.maxlen:
            frame_buffer.append(data)

        # Wait for the next frame to maintain desired FPS
        await asyncio.sleep(1 / 30)  # Assuming 30 FPS

async def transmit_frames(websocket, path=None):
    """Transmit frames from the buffer to connected WebSocket clients."""
    print("Client connected.")
    try:
        while True:
            if frame_buffer:
                # Send the latest frame
                data = frame_buffer.popleft()
                await websocket.send(data)
            else:
                pass
            await asyncio.sleep(0.01)
    except websockets.ConnectionClosed:
        print("Client disconnected.")

async def main():
    """Start the WebSocket server and frame capture."""
    server = await websockets.serve(transmit_frames, host, port)
    print(f"WebSocket server started at ws://{host}:{port}")
    await capture_and_process_frames()
    await server.wait_closed()

# Run the application
if __name__ == "__main__":
    asyncio.run(main())
