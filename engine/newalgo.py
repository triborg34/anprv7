import logging
import time
import cv2
import warnings
import torch
import asyncio
import base64
import threading
from queue import Queue
import statistics
from ultralytics import YOLO
import websockets.http
import websockets.uri
from configParams import Parameters
from database.db_entries_utils import db_entries_time
import websockets

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CCTV-Server")
logging.getLogger('torch').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Parameters
params = Parameters()
port = 5000
host = '127.0.0.1'


# Device setup
device = torch.device(0 if torch.cuda.is_available()  else "cpu")
logger.info(f"Using {'CUDA' if torch.cuda.is_available() else 'CPU'} device.")

# Frame Buffers: One buffer for each RTSP source
frame_buffers = {f"/rt{i+1}": Queue(maxsize=10) for i, _ in enumerate(params.rtps)}
global buffer_key 

# YOLO Models
class YOLOModels:
    def __init__(self, plate_model_path, char_model_path, arvand_model_path):
        logger.info("Loading YOLO models...")
        self.model_plate = torch.hub.load('yolov5', 'custom', plate_model_path, source='local', device=device, force_reload=True)
        self.model_char = torch.hub.load('yolov5', 'custom', char_model_path, source='local', force_reload=True)
        self.model_arvand = YOLO(arvand_model_path, verbose=False).to(device)
        self.carmodel=YOLO('model/yolo11n.pt',verbose=False).to(device)

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
def detect_plate_chars(cropped_plate):
    chars, confidences, char_detected = [], [], []
    results = models.model_char(cropped_plate)
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

# WebSocket Frame Transmitter
async def transmit_frames(websocket, path):
    print(path)
    """Transmit frames for the specific WebSocket endpoint."""
    logger.info(f"Client connected to {path}")
    if path not in frame_buffers.keys():
        logger.warning(f"Invalid path: {path}")
        await websocket.close()
        return

    buffer = frame_buffers[path]
    try:
        while True:
            if not buffer.empty():
                frame = buffer.get()

                # Detect plates
            # Process frame for plate detection
                car_res=models.carmodel(frame,device=device,classes=[2,7])
                if len(car_res[0])>0:
                    for box in car_res[0].boxes:
                        x1,y1,x2,y2=map(int,box.xyxy[0][:4])
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                        cropped_car=frame[y1:y2,x1:x2]
                        


           
                        plate_results = models.model_plate(cropped_car).pandas().xyxy[0]
                        
                        # plate_res=model_arvand(frame).pandas().xyxy[0]
                        # print(plate_res)
                        
                        if not plate_results.empty:
                                
                                
                            
                                for _, plate in plate_results.iterrows():
                                    plate_conf = int(plate['confidence'] * 100)
                                    if plate_conf >= int(params.plateConf):
                                        x_min, y_min, x_max, y_max = int(plate['xmin']), int(plate['ymin']), int(plate['xmax']), int(plate['ymax'])
                                        cropped_plate = cropped_car[y_min:y_max, x_min:x_max]
                                        plate_text, char_conf_avg = detect_plate_chars(cropped_plate)

                                        # Annotate frame with plate text
                                        cv2.putText(cropped_car, f"Plate: {plate_text}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.7, (0, 255, 128), 2, cv2.LINE_AA)
                                        cv2.rectangle(cropped_car,(x_min,y_min),(x_max,y_max),(0,0,255),2)
                                        plate_text.replace('Taxi','x')
                                        confidance=float(params.charConf)*100
                                        
                                
                                        # Save plate details if valid
                                        if char_conf_avg >= confidance and len(plate_text) >= 8:
                                            db_entries_time(
                                                number=plate_text,
                                                charConfAvg=char_conf_avg,
                                                plateConfAvg=plate_conf,
                                                croppedPlate=cropped_plate,
                                                status="Active",
                                                frame=frame
                                                ,isarvand='notarvand',
                                                rtpath=path
                                            )
                        
                        
                        else:                
                            plate_arvand=models.model_arvand(cropped_car,device=device)
                            models.model_arvand.to(device)
                            
                            if  len(plate_arvand[0]) >0 :
                                
                                for box in plate_arvand[0].boxes:
                                    arvand_conf=int(box.conf[0]*100)
                                    if arvand_conf >= int(params.plateConf):
                                        xMin, yMin, xMax, yMax = map(int,box[0].xyxy[0][:4])
                                        d=yMax-yMin
                                        tempyMax=yMax-int(d/2)
                                        
                                        cropped_plate_arvand = cropped_car[yMin:yMax, xMin:xMax]
                                        cropped_plate_detected_arvand = cropped_car[yMin:tempyMax, xMin:xMax]
                                        plate_text, char_conf_avg = detect_plate_chars(cropped_plate_detected_arvand)
                                        cv2.putText(cropped_car, f"Plate: {plate_text}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
                                        cv2.rectangle(cropped_car,(x_min,x_max),(y_min,y_max),(51,103,53),2)
                                        
                                        if char_conf_avg >= 60 :
                                            db_entries_time(
                                                number=plate_text,
                                                charConfAvg=char_conf_avg,
                                                plateConfAvg=plate_conf,
                                                croppedPlate=cropped_plate_arvand,
                                                status="Active",
                                                frame=frame,
                                                isarvand='arvand',
                                                rtpath=path
                                            )

                        # Encode frame as JPEG and send via WebSocket
                _, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                data = base64.b64encode(encoded).decode('utf-8')
                await websocket.send(data)
            else:
                await asyncio.sleep(0.1)  # Wait for new frames
    except websockets.ConnectionClosed:
        logger.info(f"Client disconnected from {path}")

# WebSocket Server


async def ws_handler(websocket):
    """Handle WebSocket connections."""
    path=websocket.request.path


    await transmit_frames(websocket, path)


async def websocket_server():
    """Start the WebSocket server."""
    logger.info(f"Starting WebSocket server at ws://{host}:{port}")
    
    server = await websockets.serve(
        ws_handler,
        host,
        port,
    )
    
    await asyncio.Future()  # run forever
# Main
if __name__ == "__main__":
    # Start frame producer threads for each RTSP source
    for i, source in enumerate(params.rtps):
        buffer_key = f"/rt{i+1}"
        
        threading.Thread(target=frame_producer, args=(source, frame_buffers[buffer_key]), daemon=True).start()

    # Run WebSocket server
    asyncio.run(websocket_server())
