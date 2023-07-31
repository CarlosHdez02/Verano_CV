import numpy as np
import cv2
from ultralytics import YOLO
from sort import Sort
import matplotlib.pyplot as plt
import pandas as pd

cap = cv2.VideoCapture("/Users/carloshernandez/Desktop/verano/pedestrian_dataset_1.mp4")
model = YOLO("yolov8n.pt")
tracker = Sort()
trayectorias = {}

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

    # Crear un DataFrame para almacenar las coordenadas de cada peatón
    lista_coordenadas = []
    
    # Graficar cada trayectoria y guardar las coordenadas en la lista
    for id_objeto, trayectoria in trayectorias.items():
        x = [coordenada[0] for coordenada in trayectoria]
        y = [coordenada[1] for coordenada in trayectoria]
        plt.plot(x, y, label=f"Objeto {id_objeto}")

        # Guardar las coordenadas en la lista
        for punto, coordenada in enumerate(trayectoria):
            lista_coordenadas.append({'ID_Peaton': id_objeto, 'Punto': punto + 1, 'Coordenada_X': coordenada[0], 'Coordenada_Y': coordenada[1]})

    # Convertir la lista a DataFrame
    df_trayectorias = pd.DataFrame(lista_coordenadas)
    
    # Crear listas para guardar la velocidad y la distancia de cada peatón
    lista_velocidades = []
    lista_distancia = []

    # Calcular la velocidad y la distancia para cada peatón
    for id_objeto, trayectoria in trayectorias.items():
        x = [coordenada[0] for coordenada in trayectoria]
        y = [coordenada[1] for coordenada in trayectoria]

        # Calcular la distancia total recorrida por el peatón
        distancia_total = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)

        # Calcular la velocidad promedio del peatón
        tiempo_total = len(trayectoria) - 1  # -1 para contar los pasos una vez empieza
        velocidad_promedio = distancia_total / tiempo_total

        # Guardar la velocidad y la distancia en las listas
        lista_velocidades.append({'ID_Peaton': id_objeto, 'Velocidad_promedio': velocidad_promedio})
        lista_distancia.append({'ID_peaton': id_objeto, 'Distancia_recorrida': distancia_total})

    # Convertir las listas a DataFrames
    df_velocidades = pd.DataFrame(lista_velocidades)
    df_distancia = pd.DataFrame(lista_distancia)

    # Exportar las coordenadas, velocidad y distancia a archivos CSV
    df_trayectorias.to_csv('Trayectorias.csv', index=False)
    df_velocidades.to_csv('Velocidades.csv', index=False)
    df_distancia.to_csv('Distancias.csv', index=False)

    # Mostrar la gráfica
    plt.title("Pedestrians trajectory")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()

Detector_n_Velocity(cap)
