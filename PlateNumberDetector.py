import cv2
import numpy as np

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


faceCascade = cv2.CascadeClassifier("/home/gama/Desktop/Python/OpenCV/HaarCascades/haarcascade_frontalface_default.xml")
plateCascade = cv2.CascadeClassifier("/home/gama/Desktop/Python/OpenCV/HaarCascades/haarcascade_russian_plate_number.xml")

webcam = cv2.VideoCapture(0) # En lugar de ingresar la direccion del video, un numero para el id 
webcam.set(3,300)  # Set width, id 3
webcam.set(4,300)  # Set heigth, id 4
webcam.set(10,20)  # Set brillo, id 10

index = 0

while True:
    success, img = webcam.read()
    
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(9,9),1)
    imgCanny = cv2.Canny(imgBlur,30,30)
    imgBlank = np.zeros_like(img)

    faces = faceCascade.detectMultiScale(imgGray,1.1,4)
    plates = plateCascade.detectMultiScale(imgGray,1.1,4)

    for i,face_coord in enumerate(faces):
        # print(face_coord)  # Coordenadas de la cara detectada
        x,y,wid,hei = face_coord
        imgFace = img[y:y+hei,x:x+wid]
        cv2.putText(img, "HITO "+str(i) , (x+(wid//2),y-5),
           cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 2
        )
        cv2.rectangle(img,(x,y),(x+wid,y+hei),(0,0,0),2)
        cv2.imshow("Persona",imgFace)

    for plate_coord in plates:
        x,y,wid,hei = plate_coord
        cv2.rectangle(img,(x,y),(x+wid,y+hei),(255,0,0),2)
        cv2.putText(img, "PLATE" , (x+(wid//2),y-5),
           cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 2
        )
        imgPlate = img[y:y+hei,x:x+wid]
        cv2.imshow("Plate", imgPlate)


    imgStack = stackImages(1.2,[[img,imgGray],[imgCanny,imgBlur]])

    cv2.imshow("Video",imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("/PlateNo:"+str(index)+".jpeg",imgPlate) # Path, img
        cv2.rectangle(img,(0,100),(300,200),(0,255,0),cv2.FILLED)
        cv2.putText(img,"SCAN SAVED",(120,200),cv2.FONT_HERSHEY_DUPLEX,
                    2,(0,0,255),1.5)
        cv2.imshow("Plate", imgPlate)
        cv2.waitKey(1000)
        index+=1


cv2.destroyAllWindows()

