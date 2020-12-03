import cv2
import numpy as np
import time

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


# Detectar contornos y figuras

path = "/home/gama/Pictures/shapes.jpg"
path = "/home/gama/Downloads/shapes2.jpg"

img = cv2.imread(path)
img = cv2.resize(img,(480,300))
kernel = np.ones([5,5])

def getContours(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #print(contours)

    for index,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        # cv2.drawContours(imgContour,cnt,-1,(255,0,0),2) #img,cnt,cntidx,color,thickness
        #if area > 500 and area < 5000:
        if area > 0:
            # Dibujar el contorno
            print(area)
            cv2.drawContours(imgContour,cnt,-1,(255,255,0),2) #img,cnt,cntidx,color,thickness
            
            perimetro = cv2.arcLength(cnt,True)
            #print(perimetro)
            aprox = cv2.approxPolyDP(cnt,0.02*perimetro,True)
            objCor = len(aprox)   # La cantidad de aristas ???
            print(objCor)        # Coordenadas del objeto     
            
            # Encerrar en rectangulos las imagenes
            x,y,wid,hei = cv2.boundingRect(aprox)
            cv2.rectangle(imgContour,(x,y),(x+wid,y+hei),(0,0,0),2)
            
            # Contar vertices, y detectar figuras
            if objCor == 3:   # Triangulo
                tipo = "Tri"
            elif objCor == 4:
                if abs(wid - hei) > 10:
                    tipo = "Rect"    
                else:
                    tipo = "Cuad"
            elif objCor > 4:
                tipo = "Circle"
            # Musho Texto
            cv2.putText(imgContour,tipo, (x+(wid//2),y+(hei//2)),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2
            )
            cv2.putText(imgContour,str(index), (x+(wid//2),y+(hei//2)+20),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2
            )


imgContour = np.copy(img)
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny = cv2.Canny(imgBlur,60,60)
imgBlank = np.zeros_like(img)

getContours(imgCanny)

imgStack = stackImages(1.2,[[img,imgGray,imgBlur],[imgCanny,imgContour,imgBlank]])

while True:

    cv2.imshow("Resultados",imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
