import cv2

video = cv2.VideoCapture()#Escribe nombre del vidio

def upgrade_video_quality(video):
    capture = cv2.VideoCapture(video) # Capturando el video
    

    # Obtener las dimensiones del video
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Establecer el formato del video de salida
    video_format = cv2.VideoWriter_fourcc(*'mp4v')

    # Crear el objeto de video para guardar el video filtrado
    output_video = cv2.VideoWriter('FilteredVideo.mp4', video_format, 30.0, (width, height))

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        # Aplicar el filtro Gaussiano al frame
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        median = cv2.medianBlur(frame,5)

        #Muestro los 2 para compararlos
        cv2.imshow('Original', frame)
        cv2.imshow('Filtrado', median)
        
        output_video.write(blurred)

        key = cv2.waitKey(1)
        if key == 27:
            break

    capture.release()
    output_video.release()
    cv2.destroyAllWindows()

upgrade_video_quality(video)
