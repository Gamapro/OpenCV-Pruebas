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

width, height = 300,400
BLANCO = (255,255,255)

webcam = cv2.VideoCapture(0) # En lugar de ingresar la direccion del video, un numero para el id
webcam.set(3,600)  # Set width, id 3
webcam.set(4,600)  # Set heigth, id 4
webcam.set(10,100)  # Set brillo, id 10

colores = [
    ["AZUL",56,129,85,205,67,255,(64,62,247)],      # AZUL REY
    ["AMARILLO",24,255,45,255,180,255,(255,240,5)], # AMARILLO FLUORESCENTE
    ["ROSA",28,255,121,255,0,255,(255,65,255)]      # ROSA
    #["NARANJA",5,107,0,19,255,255],
    #["VERDE",57,76,0,100,255,255]
]

def getContours(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,wid,hei = 0,0,0,0
    for index,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        # cv2.drawContours(imgContour,cnt,-1,(255,0,0),2) #img,cnt,cntidx,color,thickness
        if area > 400:
            # Dibujar el contorno
            cv2.drawContours(imgCopy,cnt,-1,(255,255,0),2) #img,cnt,cntidx,color,thickness
            perimetro = cv2.arcLength(cnt,True)
            aprox = cv2.approxPolyDP(cnt,0.02*perimetro,True)
            objCor = len(aprox)   # La cantidad de aristas ???
            # Encerrar en rectangulos las imagenes
            x,y,wid,hei = cv2.boundingRect(aprox)
            #cv2.rectangle(imgCopy,(x,y),(x+wid,y+hei),(0,0,0),2)
    return x+(wid//2), y

points = []

def drawPoints(img):
    for point in points:
        cv2.circle(img,point[:2],6,point[2],cv2.FILLED)

    return img

while True:

    success, img = webcam.read()
    #cv2.imshow("Video",img)

    img = cv2.resize(img,(width,height))   # Widht, Heigth
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    imgCopy = np.copy(img)
    
    frames = [imgCopy]

    for index,color in enumerate(colores):
        if index > 2:
            break
        lw = np.array(color[1:7:2])
        up = np.array(color[2:7:2])
        mask = cv2.inRange(imgHSV,lw,up)
        x,y = getContours(mask)
        if not (x,y,color[-1]) in points:
            points.append((x,y,color[-1]))
        cv2.putText(mask,color[0],(width//2,height-40),cv2.FONT_ITALIC,0.5,BLANCO,2)
        cv2.circle(imgCopy,(x,y),10,color[-1],cv2.FILLED)
        frames.append(mask)
    
    imgCopy = drawPoints(imgCopy)

    #imgStack = stackImages(1,[frames[:3],frames[3:]])
    imgStack = stackImages(1,[frames[:2],frames[2:4]])

    cv2.imshow("Resultados",imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        points = []

cv2.destroyAllWindows()    # Limpiar las ventanas



