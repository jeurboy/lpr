import re
import time
from datetime import datetime
import cv2
import imutils
import numpy as np
import pytesseract
# from PIL import Image
import matplotlib.pyplot as plt


# apt-get install tesseract-ocr-LANG
# pip3 install numpy tesseract opencv-python imutils matplotlib
# brew install tesseract tesseract-lang tesseract-ocr-tha

kernel_sharp = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]])

scale = 0.5
pattern = "^[0-9]?[ก-ฮ]{1,2}\s+[0-9]{1,4}[^0-9]+\s+[ก-๏]+"


def detect_plate(img):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Blur to reduce noise

    edged = cv2.Canny(gray, 30, 200)  # Perform Edge detection
    # cv2.imshow('gray', gray)

    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour

    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    screenCnt = None

    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0

        return
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 10)

    # Masking the part other than the number plate
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [screenCnt], 0, 255, -1,)

    new_image = cv2.bitwise_and(img, img, mask=mask)
    # new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    # Cropped = squarize(mask, screenCnt)

    # Now crop
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx-1, topy:bottomy-1]

    Cropped = squarize(new_image, screenCnt)
    # Cropped = squarize(gray, screenCnt)    # Sharpen image
    # Cropped = cv2.filter2D(src=Cropped, ddepth=-1, kernel=kernel_sharp)

    # Resize crop impage
    Cropped = cv2.resize(Cropped, (round(260), round(120)),
                         interpolation=cv2.INTER_LINEAR)

    cv2.imshow('Cropped', Cropped)
    # Read the number plate
    # for num in ["3", "6", "10", "11", "12", "13"]:

    for num in ["11", "13"]:
        text = pytesseract.image_to_string(
            Cropped, lang='tha', config='--dpi 2400 --oem 1 --psm '+num)

        if text != '' and re.search(pattern, text):
            print(current_time + ": Detected Number["+num+"] is \n", text)
            cv2.imshow("Cropped", Cropped)

            time.sleep(1)


def squarize(img, screenCnt):
    width = round(1920 * scale)
    height = round(1080 * scale)
    screenCnt.sort(axis=1)

    # (x, y) = np.where(screenCnt)
    # minx = np.amin(screenCnt)
    # (topx, topy) = (np.min(x), np.min(y))
    # (bottomx, bottomy) = (np.max(x), np.max(y))

    print(screenCnt[0][0])
    print(screenCnt[1][0])
    print(screenCnt[2][0])
    print(screenCnt[3][0])

    print('===================')
    time.sleep(0.3)

    # pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts1 = np.float32([screenCnt[1][0], screenCnt[0][0],
                      screenCnt[2][0], screenCnt[3][0]])
    pts2 = np.float32(
        [[0, 0], [width+1, 0], [0, height+1], [width+1, height+1]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (width, height))

    # plt.subplot(121), plt.imshow(img), plt.title('Input')
    # plt.subplot(122), plt.imshow(dst), plt.title('Output')
    # plt.show()
    # cv2.imshow('img 1', img)
    cv2.imshow('dst 1', dst)

    return dst


# print(pytesseract.get_languages())
# set up the camera
vidcap = cv2.VideoCapture(0)

# loop through frames from camera
while True:
    if vidcap.isOpened():
        ret, frame = vidcap.read()  # capture a frame from live video

    # check whether frame is successfully captured
        if ret:
            frame = cv2.resize(frame, (round(1920 * scale), round(1080*scale)))
            # frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel_sharp)

            cv2.imshow("img", frame)  # show captured frame
            detect_plate(frame)

            # time.sleep(0.25)

            # press 'q' to break out of the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # print error if frame capturing was unsuccessful
    else:
        print("Error : Failed to capture frame")

# print error if the connection with camera is unsuccessful
else:
    print("Cannot open camera")

    # time.sleep(1)

# release the camera and close windows

vidcap.release()
cv2.destroyAllWindows()
