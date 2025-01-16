import logging
import statistics
import warnings
import torch
from ultralytics import YOLO
import cv2
from configParams import Parameters

logging.getLogger('torch').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('ultralytics').setLevel(logging.ERROR)
params = Parameters()


def getDevice():
    if torch.cuda.is_available():
        return 0
    elif torch.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    
device=getDevice()




model_plate = torch.hub.load('yolov5', 'custom', params.modelPlate_path, source='local', device=device, force_reload=True)
model_char = torch.hub.load('yolov5', 'custom', params.modelCharX_path, source='local', device=device, force_reload=True)
model_arvand=YOLO(params.modelArvand_path,verbose=False)
modelcar=YOLO('model/yolo11n.pt')
modelcar.to(device)



cap=cv2.VideoCapture('rtsp://admin:admin@192.168.1.89:554/mainstream')




def detect_plate_chars(cropped_plate):
    chars, confidences, char_detected = [], [], []
    results = model_char(cropped_plate)
    detections = sorted(results.pred[0], key=lambda x: x[0])  # Sort by x-coordinate
    for det in detections:
        conf = det[4]
        confidance=int(params.charConf)/100
        if conf > confidance:
            cls = int(det[5].item())
            char = params.char_id_dict.get(str(cls), '')
            chars.append(char)
            confidences.append(conf.item())
            char_detected.append(det.tolist())
    char_conf_avg = round(statistics.mean(confidences) * 100) if confidences else 0
    return ''.join(chars), char_conf_avg



while True:
        ret,frame=cap.read()

        if not ret:
            break
        res_car=modelcar(frame,device=device,classes=[2])
        if len(res_car[0])>0:
            for box in res_car[0].boxes:
                x1,y1,x2,y2=map(int,box.xyxy[0][:4])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),5)
                cropped_car=frame[y1:y2,x1:x2]
                plate_results = model_plate(cropped_car).pandas().xyxy[0]
                
                if not plate_results.empty:
                    for _,plate in plate_results.iterrows():
                        plate_conf = int(plate['confidence'] * 100)
                        
                        if plate_conf >=60:
                            x_min, y_min, x_max, y_max = int(plate['xmin']), int(plate['ymin']), int(plate['xmax']), int(plate['ymax'])
                            cropped_plate = cropped_car[y_min:y_max, x_min:x_max]
                            cv2.rectangle(cropped_car,(x_min,y_min),(x_max,y_max),(255,255,255),5)
                            plate_text,conf=detect_plate_chars(cropped_plate)
                            print(plate_text+'\n'+str(conf))
        cv2.imshow('fr',frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
                break




# cap.release()

# cv2.destroyAllWindows()

                            



