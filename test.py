import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image

# apt-get install tesseract-ocr-LANG
# pip3 install numpy tesseract opencv-python imutils
# brew install tesseract tesseract-lang tesseract-ocr-tha

kernel_sharp = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]])

# print(pytesseract.get_languages())

img = cv2.imread('1.jpeg', cv2.IMREAD_COLOR)
# img = cv2.resize(img, (620, 480))
# img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel_sharp)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Blur to reduce noise

edged = cv2.Canny(gray, 30, 200)  # Perform Edge detection
# cv2.imshow('edged', edged)

# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour

cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    print("No contour detected")
    exit()
else:
    detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 10)

# Masking the part other than the number plate
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1,)
new_image = cv2.bitwise_and(img, img, mask=mask)

# cv2.imshow('new_image', new_image)
# cv2.imshow('mask', mask)
# Now crop
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

# Cropped = cv2.filter2D(src=Cropped, ddepth=-1, kernel=kernel_sharp)

# Read the number plate
for num in ["3", "6", "10", "11", "12", "13"]:
    # text = pytesseract.image_to_string(
    #     Cropped, lang='tha+eng', config='--dpi 2400 --oem 1 --psm '+num)
    text = pytesseract.image_to_string(
        Cropped, lang='tha+script/Thai', config='--dpi 2400 --oem 1 --psm '+num)
    print("Detected Number ["+num+"] is:", text)

cv2.imshow('image', img)
cv2.imshow('Cropped', Cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
