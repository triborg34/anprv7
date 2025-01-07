from ultralytics import YOLO
import cv2
import statistics
import torch

# import easyocr

# reader =easyocr.Reader(['en'], gpu=True)

# char_dict = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8',
#                           '9': '9',
# }

# char_id_dict = {v: k for k, v in char_dict.items()}
 
# def detect_plate_chars(cropped_plate):
#     chars, confidences, char_detected = [], [], []
#     results = model_char(cropped_plate)
#     detections = sorted(results.pred[0], key=lambda x: x[0])  # Sort by x-coordinate
#     for det in detections:
#         conf = det[4]
#         confidance=0.5
#         if conf > confidance:
#             cls = int(det[5].item())
#             char = char_id_dict.get(str(cls), '')
#             chars.append(char)
#             confidences.append(conf.item())
#             char_detected.append(det.tolist())
#     char_conf_avg = round(statistics.mean(confidences) * 100) if confidences else 0
#     return ''.join(chars), char_conf_avg

# model_char = torch.hub.load('yolov5', 'custom', 'model/CharsYolo.pt', source='local', device=0, force_reload=True)
# model=YOLO("model/arvand.pt")
# frame=cv2.imread("319.jpg")
# res=model(frame)

# if len(res[0]) > 0:
#     for box in res[0].boxes:
#         x_min, y_min, x_max, y_max = map(int, box.xyxy[0][:4])
#         cropped_plate = frame[y_min:y_max, x_min:x_max]
#         plate_text, char_conf_avg = detect_plate_chars(cropped_plate)
#         cv2.imwrite('cropped.jpg', cropped_plate)
#         text=reader.readtext(cropped_plate,detail=0)
#         print(text)
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

#         cv2.imshow("cropped", frame)
#         cv2.waitKey(0)



modelcar=YOLO('model/yolo11n.pt')
frame=cv2.imread('22.png')
res=modelcar(frame)
print(modelcar.names)
if len(res[0]) > 0:
        annotated_frame = res[0].plot()
        cv2.imshow('frame', annotated_frame)
        cv2.waitKey(0)