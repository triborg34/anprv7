import logging
import cv2
import torch
import time
import statistics
import warnings
import asyncio
import base64
import queue
import threading
import websockets
from configParams import Parameters
from database.db_entries_utils import db_entries_time



logging.getLogger('torch').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
params = Parameters()

# WebSocket parameters
port = 5000
host = '127.0.0.1'

# Device setup
def get_device():
    if torch.cuda.is_available():
        print("Using CUDA device.")
        return 0
    else:
        print("Using CPU device.")
        return torch.device("cpu")

device = get_device()

# Load YOLO models
model_plate = torch.hub.load('yolov5', 'custom', params.modelPlate_path, source='local', device=device, force_reload=True)
model_char = torch.hub.load('yolov5', 'custom', params.modelCharX_path, source='local', device=device, force_reload=True)

# RTSP or video source setup
source = params.rtps if params.rtps else 0  # Use RTSP if defined; otherwise, default to webcam
cap = cv2.VideoCapture(source)

# Initialize variables for buffer logic
buffer_logic = str(device) == "cpu"  # Use buffer only when running on CPU

buffer_logic=True
print(buffer_logic)

frame_buffer = queue.Queue(maxsize=5) if buffer_logic else None  # Only create buffer if on CPU
retry_count = 0
max_retries = 5

# Function to fill the buffer asynchronously (only used on CPU)
def buffer_filler():
    global retry_count
    while retry_count < max_retries:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not retrieve frame. Retrying ({retry_count + 1}/{max_retries})...")
            retry_count += 1
            time.sleep(1)  # Retry delay
            continue

        retry_count = 0  # Reset retry count on successful frame read

        if frame_buffer.full():
            frame_buffer.get()  # Remove the oldest frame if the buffer is full

        frame_buffer.put(frame)

# Start buffer filler thread if using CPU
if buffer_logic:
    print("Buffer logic enabled (CPU mode).")
    threading.Thread(target=buffer_filler, daemon=True).start()

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

# WebSocket frame transmitter
async def transmit_frames(websocket, path=None):
    """Transmit frames to WebSocket clients."""
    print("Client connected.")
    try:
        while True:
            if buffer_logic:
                # CPU: Use buffer to get frames
                if not frame_buffer.empty():
                    frame = frame_buffer.get()
                else:
                    await asyncio.sleep(0.1)  # Delay before checking buffer again
                    continue
            else:
                # CUDA: Read frame directly from the source
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not retrieve frame.")
                    await asyncio.sleep(1)  # Retry delay
                    continue

            # Process frame for plate detection
            plate_results = model_plate(frame).pandas().xyxy[0]
            for _, plate in plate_results.iterrows():
                plate_conf = int(plate['confidence'] * 100)
                if plate_conf >= 85:
                    x_min, y_min, x_max, y_max = int(plate['xmin']), int(plate['ymin']), int(plate['xmax']), int(plate['ymax'])
                    cropped_plate = frame[y_min:y_max, x_min:x_max]
                    plate_text, char_conf_avg = detect_plate_chars(cropped_plate)

                    # Annotate frame with plate text
                    cv2.putText(frame, f"Plate: {plate_text}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    plate_text.replace('Taxi','x')
                    # Save plate details if valid
                    if char_conf_avg >= 75 and len(plate_text) >= 8:
                        db_entries_time(
                            number=plate_text,
                            charConfAvg=char_conf_avg,
                            plateConfAvg=plate_conf,
                            croppedPlate=cropped_plate,
                            status="Active",
                            frame=frame
                        )

            # Encode frame as JPEG and send via WebSocket
            _, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            data = base64.b64encode(encoded).decode('utf-8')
            await websocket.send(data)
    except websockets.ConnectionClosed:
        print("Client disconnected.")

# Main WebSocket server function
async def main():
    """Start the WebSocket server."""
    server = await websockets.serve(transmit_frames, host, port)
    print(f"WebSocket server started at ws://{host}:{port}")
    await server.wait_closed()

# Run the application
if __name__ == "__main__":
    asyncio.run(main())
    cap.release()
