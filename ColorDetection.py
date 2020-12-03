import cv2
import numpy as np
import time

# TRACKBARS

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


img = cv2.imread("/home/gama/Downloads/cosos/pato.jpeg")
width, height = 300,400

def mi_funcion(val):
    return

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars",300,20)

cv2.createTrackbar("Hue_Min","Trackbars",0,255,mi_funcion)
cv2.createTrackbar("Hue_Max","Trackbars",100,255,mi_funcion)
cv2.createTrackbar("Sat_Min","Trackbars",0,255,mi_funcion)
cv2.createTrackbar("Sat_Max","Trackbars",200,255,mi_funcion)
cv2.createTrackbar("Val_Min","Trackbars",20,255,mi_funcion)
cv2.createTrackbar("Val_Max","Trackbars",255,255,mi_funcion)

while True:

    #img = cv2.imread("/home/gama/Downloads/cosos/pato.jpeg")
    img = cv2.resize(img,(width,height))   # Widht, Heigth
    
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
   
    h_min = cv2.getTrackbarPos("Hue_Min","Trackbars")
    h_max = cv2.getTrackbarPos("Hue_Max","Trackbars")
    s_min = cv2.getTrackbarPos("Sat_Min","Trackbars")
    s_max = cv2.getTrackbarPos("Sat_Max","Trackbars")
    v_min = cv2.getTrackbarPos("Val_Min","Trackbars")
    v_max = cv2.getTrackbarPos("Val_Max","Trackbars")
    
    #print((h_min,h_max,s_min,s_max,v_min,v_max))
    
    lw = np.array([h_min,s_min,v_min])
    up = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lw,up)

    mask = mask[...,np.newaxis]
    mask = np.concatenate([mask,mask,mask],axis=2)
    mask2 = np.copy(mask)

    result = cv2.bitwise_and(mask,img,mask)
    #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    imgStack = stackImages(1.2,[[img,mask2],[imgHSV,result]]) #np.hstack([img,imgHSV,mask2,result])

    cv2.imshow("Resultados",imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
