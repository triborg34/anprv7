import logging
import cv2
import torch
import asyncio
import base64
import threading
from queue import Queue
from ultralytics import YOLO
from configParams import Parameters
from database.db_entries_utils import db_entries_time
import websockets

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CCTV-Server")

# Parameters
params = Parameters()
port = 5000
host = '127.0.0.1'

# Device setup
device = torch.device(0 if torch.cuda.is_available() else "cpu")
logger.info(f"Using {'CUDA' if torch.cuda.is_available() else 'CPU'} device.")

# Frame Buffers: One buffer for each RTSP source
frame_buffers = {source: Queue(maxsize=10) for source in params.rtps}

# YOLO Models
class YOLOModels:
    def __init__(self, plate_model_path, char_model_path, arvand_model_path):
        logger.info("Loading YOLO models...")
        self.model_plate = torch.hub.load('yolov5', 'custom', plate_model_path, source='local', device=device, force_reload=True)
        self.model_char = torch.hub.load('yolov5', 'custom', char_model_path, source='local', force_reload=True)
        self.model_arvand = YOLO(arvand_model_path, verbose=False).to(device)

models = YOLOModels(params.modelPlate_path, params.modelCharX_path, params.modelArvand_path)

# Frame Producer
def frame_producer(source, buffer):
    """Capture frames from an RTSP source and add them to a buffer."""
    cap = cv2.VideoCapture(source)
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error(f"Failed to retrieve frame from {source}. Retrying...")
            continue
        if buffer.full():
            buffer.get()  # Remove oldest frame if buffer is full
        buffer.put(frame)

# Character Detection
def detect_plate_chars(cropped_plate, model_char):
    """Detect characters on the license plate."""
    results = model_char(cropped_plate)
    chars, confidences = [], []
    for det in results.pandas().xyxy[0].iterrows():
        conf = det[1]['confidence']
        cls = int(det[1]['class'])
        if conf > float(params.charConf) / 100:
            chars.append(params.char_id_dict.get(str(cls), ''))
            confidences.append(conf)
    char_conf_avg = round(sum(confidences) / len(confidences) * 100) if confidences else 0
    return ''.join(chars), char_conf_avg

# WebSocket Frame Transmitter
async def transmit_frames(websocket, path):
    logger.info("Client connected.")
    try:
        while True:
            # Iterate through buffers and fetch frames
            for source, buffer in frame_buffers.items():
                if not buffer.empty():
                    frame = buffer.get()

                    # Detect plates
                    plate_results = models.model_plate(frame).pandas().xyxy[0]
                    for _, plate in plate_results.iterrows():
                        conf = int(plate['confidence'] * 100)
                        if conf >= int(params.plateConf):
                            x_min, y_min, x_max, y_max = map(int, [plate['xmin'], plate['ymin'], plate['xmax'], plate['ymax']])
                            cropped_plate = frame[y_min:y_max, x_min:x_max]
                            plate_text, char_conf_avg = detect_plate_chars(cropped_plate, models.model_char)

                            if char_conf_avg >= 75 and len(plate_text) >= 8:
                                db_entries_time(
                                    number=plate_text,
                                    charConfAvg=char_conf_avg,
                                    plateConfAvg=conf,
                                    croppedPlate=cropped_plate,
                                    status="Active",
                                    frame=frame
                                )

                    # YOLO Arvand Model
                    arvand_results = models.model_arvand(frame)
                    for box in arvand_results[0].boxes:
                        conf = box.conf[0]
                        if conf >= int(params.plateConf):
                            x_min, y_min, x_max, y_max = map(int, box.xyxy[0][:4])
                            cropped_plate = frame[y_min:y_max, x_min:x_max]
                            plate_text, char_conf_avg = detect_plate_chars(cropped_plate, models.model_char)

                            if char_conf_avg >= 75:
                                db_entries_time(
                                    number=plate_text,
                                    charConfAvg=char_conf_avg,
                                    plateConfAvg=conf,
                                    croppedPlate=cropped_plate,
                                    status="Active",
                                    frame=frame
                                )

                    # Encode frame and send
                    _, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    await websocket.send(base64.b64encode(encoded).decode('utf-8'))
            await asyncio.sleep(0.1)  # Prevent CPU overutilization
    except websockets.ConnectionClosed:
        logger.info("Client disconnected.")

# WebSocket Server
async def websocket_server():
    """Start the WebSocket server."""
    logger.info(f"Starting WebSocket server at ws://{host}:{port}")
    async with websockets.serve(transmit_frames, host, port):
        await asyncio.Future()  # Run forever

# Main
if __name__ == "__main__":
    # Start frame producer threads for each RTSP source
    for source, buffer in frame_buffers.items():
        threading.Thread(target=frame_producer, args=(source, buffer), daemon=True).start()

    # Run WebSocket server
    asyncio.run(websocket_server())
