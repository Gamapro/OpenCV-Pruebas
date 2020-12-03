import cv2
import numpy as np

# Leer imagenes, videos y webcams

# Imagen
img = cv2.imread("/home/gama/Downloads/cosos/pato.jpeg")   # imgRead
cv2.imshow("IMAGEN",img)    # imgShow con 2 args: Nombre de la ventana, nombre del archivo a mostrar
cv2.waitKey(0)              # Delay, cuando tiempo en ms se mostrara en pantalla, 0 lo muestra indefinidamente

# Video
video = cv2.VideoCapture("/home/gama/Downloads/cosos/"+
                        "Meet - kbq-qpfd-oav - Google Chrome 18_03_2020 12_31_33 p. m..mp4")

while True:
    success, img = video.read()
    cv2.waitKey(1)
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()            # Liberar la camara
cv2.destroyAllWindows()    # Limpiar las ventanas

# Webcam
webcam = cv2.VideoCapture(0) # En lugar de ingresar la direccion del video, un numero para el id

webcam.set(3,600)  # Set width, id 3
webcam.set(4,600)  # Set heigth, id 4
webcam.set(10,100)  # Set brillo, id 10

while True:
    success, img = webcam.read()
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        breakq

webcam.release()           # Liberar la webcam
cv2.destroyAllWindows()    # Limpiar las ventanas
