import cv2
import math
import numpy as np


def nothing(x):
    pass


font = cv2.FONT_HERSHEY_COMPLEX
cv2.namedWindow("Adjust")
cv2.createTrackbar("min", "Adjust", 110, 255, nothing)


def getcontours(vdo, th1):
    contours, hierarchy = cv2.findContours(vdo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for ix, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area > 1000:
            cv2.drawContours(th1, contours, -1, (0, 255, 0), 3)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            objcor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            # Определение точек и угла:

            if objcor == 7:
                cv2.rectangle(th1, (x, y), (x + w, y + h), (255, 0, 0), 2)
                startpoint = (approx[0][0][0], approx[0][0][1])
                endpoint = (int((approx[3][0][0] + approx[4][0][0]) / 2), int((approx[3][0][1] + approx[4][0][1]) / 2))
                slope = (startpoint[0] - endpoint[0]) / (startpoint[1] - endpoint[1])
                angle = math.degrees(math.atan2(startpoint[0] - endpoint[0], startpoint[1] - endpoint[1]))
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(th1, str(angle), (10, 85 + 50 * ix), font, 1, (255, 255, 0))
                print('angle=', angle)
            else:
                pass


while True:
    img = cv2.imread('1.jpg')
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mi = cv2.getTrackbarPos("min", "Adjust")

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([mi, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    result = cv2.bitwise_and(img, img, mask=mask)

    ab, th1 = cv2.threshold(result, 133, 255, cv2.THRESH_BINARY)
    th1 = cv2.erode(th1, None, iterations=2)
    th1 = cv2.dilate(th1, None, iterations=2)

    getcontours(mask, th1)

    cv2.imshow('th1', th1)
    cv2.imshow("frame", img)

    key = cv2.waitKey(500)

    if key == 27:
        break


