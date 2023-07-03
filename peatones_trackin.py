import cv2
import numpy as np
from ultralytics import YOLO
#from sort import Sort 

cap = cv2.VideoCapture('/Users/carloshernandez/Desktop/verano/pedestrian_dataset_1.mp4')
model = YOLO('yolov8m.pt') #Calling the YOLO CNN

#tracker = Sort() #inicializar objeto 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
   
    #results = model(frame,stream=True)
    

    '''for res in results:
        boxes = res.boxes.xyxy.cpu().numpy().satyp(int) 
        tracks = tracker.update(boxes)
        tracks = tracks.astype(int) #Transformando a int
        #print(tracks) #Se generan bboxes y ids
        #print(res.boxes)

        for xmin,ymin,xmax,ymax, track_id in tracks:
            cv2.putText(img=frame, text=f"Id: {track_id}", org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,255,0), thickness=2)
            cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)

        break
    '''
    #Aqui habia un break que rompia el codigo de abajo

    frame = results[0].plot() #Aqui grafico las bboxes

    cv2.imshow('dataset',frame) #Mostrando el vidio
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()