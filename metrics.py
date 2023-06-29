import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pedestrians_detection import show_dataset


cap = cv2.VideoCapture('video path here') #Video input
model = YOLO('yolov8m.pt') #Calling the YOLO CNN
fig,axs = plt.subplots(2,2) #Creando la figura de 2x2 para ver las metricas graficadas


show_dataset(cap)