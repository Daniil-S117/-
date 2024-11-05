import imutils
import cv2
import keyboard
import numpy as np
from skimage.morphology import skeletonize


def main():
    flag = True
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=700)
        if flag:
            frame = Detector(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(30) & keyboard.is_pressed("SPACE"):
            cv2.waitKey(-1)
    else:
        print('Error read the image!!')
    cap.release()
    cv2.destroyAllWindows()


def Detector(_frame):
    ## Read image file.
    img = _frame

    ## Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = ~gray_img
    ## Improve contrast in image
    ## Using hitogram equalization
    contrast_img = cv2.equalizeHist(gray_img)

    ## Aply binary thresholding
    ret, thresh1 = cv2.threshold(contrast_img, 230, 255, cv2.THRESH_BINARY)

    '''Different thresholding can e implemented. Pick one from below. 
	# ret,thresh1_sk = cv2.threshold(contrast_img,140,1,cv2.THRESH_BINARY_INV) 
	# mean_t = cv2.adaptiveThreshold(contrast_img,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2)
	# gaussian_t = cv2.adaptiveThreshold(contrast_img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_IV,11,2)
	'''

    ## Skeletonization
    ## Convert to binary 0 and 1
    ske = (skeletonize(thresh1 / 255) * 255).astype(np.uint8)

    ## Detect edges from the skeleton image
    edges = cv2.Canny(ske, 100, 200)

    ## Apply Hough Transform
    ## This returns an array of r and theta values
    lines = cv2.HoughLines(ske, 1, np.pi / 180, 50)

    ## Draw the lines
    for rho, theta in lines[0]:
        print(rho, theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 - 1000 * (-b))
        y1 = int(y0 - 1000 * (a))
        x2 = int(x0 + 1000 * (-b))
        y2 = int(y0 + 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (37, 253, 13), 2)

    ## Show Hough transformed images
    res = np.hstack((contrast_img, thresh1, ske, edges))
    #cv2.imshow("res.png", res)
    #cv2.imshow("final.png", img)
    return img


if __name__ == '__main__':
    main()
