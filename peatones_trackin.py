import numpy as np
import cv2
from ultralytics import YOLO
from sort import Sort
import matplotlib.pyplot as plt
import pandas as pd

cap = cv2.VideoCapture("/Users/carloshernandez/Desktop/verano/pedestrian_dataset_1.mp4") #Escogiendo el video
#Guardar coordenadas
#Guardar en un CSV las coordenadas de cada peaton
model = YOLO("yolov8n.pt") #Llamando al modelo
tracker = Sort() #inicializando el objeto
trayectorias = {} #Se guarda en un diccionario donde cada objeto es una llave y la trayectoria el valor

def Detector_n_Velocity(video):
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, stream=True)

        for res in results:
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.5)[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            tracks = tracker.update(boxes)
            tracks = tracks.astype(int)

            for xmin, ymin, xmax, ymax, track_id in tracks:
                if track_id not in trayectorias:
                    trayectorias[track_id] = []
                
                coordenadas = [(xmin + xmax) // 2, (ymin + ymax) // 2]
                trayectorias[track_id].append(coordenadas)

                cv2.putText(img=frame, text=f"Id: {track_id}", org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,255,0), thickness=2)
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    # Making the picture
    plt.title("Pedestrians trajectory")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.grid()

    lista_velocidades = [] #Creo un vector para guardar las velocidades

    # Graficar cada trayectoria
    for id_objeto, trayectoria in trayectorias.items():
        x = [coordenada[0] for coordenada in trayectoria]
        y = [coordenada[1] for coordenada in trayectoria]
        plt.plot(x, y, label=f"Objeto {id_objeto}")

        # Calcular la velocidad promedio del peaton con el ID id_objeto
        tiempo = np.arange(len(trayectoria))  #Se guarda todo el contenido de trayectoria
        x = np.array(x) #Se crea un vector para la posicion x
        y = np.array(y) #Se crea un vector para la posicion y

        # distancia total recorrida
        distancia_total = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)

        # Calcula el tiempo, se usa len para encontrar la longitud de la trayectoria
        tiempo_total = len(trayectoria) - 1 #-1 para contar los pasos una vez empieza

        # Calcula la velocidad promedio
        velocidad_promedio = distancia_total / tiempo_total
        print(f"Velocidad promedio para objeto {id_objeto}: {velocidad_promedio}")
        
        #Ahora lo guardo en un CSV
        lista_velocidades.append({'ID_Peaton': id_objeto, 'Velocidad_promedio': velocidad_promedio})


        df = pd.DataFrame(lista_velocidades)

        df.to_csv('Velocidades2.csv', index=False)

    # Showing the picture
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()


#Now in this function what i do is plot the trajectories
def show_trajectories(video):
    while cap.isOpened():
        ret, frame = cap.read() #Reading each frame of the video

        if not ret: #If there is no video, break the loop
            break

        results = model(frame, stream=True)

        for res in results:
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.5)[0] #Filtro los indices para tener los mayores a 0.5
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            tracks = tracker.update(boxes)
            tracks = tracks.astype(int)

            for xmin, ymin, xmax, ymax, track_id in tracks: #Coordenadas para las bboxes
                if track_id not in trayectorias:
                    trayectorias[track_id] = [] #Hago el array para guardar los Ids
                
                coordenadas = [(xmin + xmax) // 2, (ymin + ymax) // 2]  # Coordenadas del centro del objeto
                trayectorias[track_id].append(coordenadas) #Agrego el ID a coordenadas

                cv2.putText(img=frame, text=f"Id: {track_id}", org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,255,0), thickness=2)
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)
                

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    # Making the picture
    plt.title("Pedestrians trajectory")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.grid()
    # Graficar cada trayectoria
    for id_objeto, trayectoria in trayectorias.items():
        x = [coordenada[0] for coordenada in trayectoria]  # Coordenadas X
        y = [coordenada[1] for coordenada in trayectoria]  # Coordenadas Y
        plt.plot(x, y, label=f"Objeto {id_objeto}")
        

    #Showing the picture
    plt.legend(bbox_to_anchor = (1,1),loc='upper left')
    plt.show()

#In this function i leave out the ground truth and put an ID in all of the objects to track them using the sort tracking algorithm
def detector_n_tracker(video):
    #Sort es muy sensible a ruido
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
                break

        results = model(frame, stream=True)

        for res in results:
                filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.5)[0] #Que detecte cuando la confidencialidad es mayor a 0.5
                boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int) #Muestras de tipo int 
                tracks = tracker.update(boxes)
                tracks = tracks.astype(int)
                
                for xmin, ymin, xmax, ymax, track_id in tracks: #Dibujando el rectangulo

                    cv2.putText(img=frame, text=f"Id: {track_id}", org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,255,0), thickness=2)
                    cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)
            
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    
#In this function i detect the objects and show the ground truth value of the detector that goes from 0-1
def ground_truth(video):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        frame = results[0].plot() #Aqui grafico las bboxes
        cv2.imshow('dataset',frame) #Mostrando el vidio
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#Here i call the functions

Detector_n_Velocity(cap)
#show_trajectories(cap)
#detector_n_tracker(cap)
#ground_truth(cap)

#Subir yolov8n

