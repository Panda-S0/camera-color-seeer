
import cv2
import numpy as np
import time

# img_path= "imgs/5Fingers.jpg"
# img = cv2.imread(img_path)
# cv2.imshow('Fingers',img)
# cv2.waitKey(0)
def empty(a):
    pass
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",600,600)
cv2.createTrackbar("Hue Min", "HSV", 0, 179,empty)
cv2.createTrackbar("Hue Max", "HSV", 179, 179,empty)
cv2.createTrackbar("H In Out", "HSV", 0, 1,empty)
cv2.createTrackbar("Sat Min", "HSV", 0, 255,empty)
cv2.createTrackbar("Sat Max", "HSV", 255, 255,empty)
cv2.createTrackbar("S In Out", "HSV", 0, 1,empty)
cv2.createTrackbar("Val Min", "HSV", 0, 255,empty)
cv2.createTrackbar("Val Max", "HSV", 255, 255,empty)
cv2.createTrackbar("V In Out", "HSV", 0, 1,empty)
cv2.createTrackbar("min squar size", "HSV", 800, 10000,empty,)

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

cv2.namedWindow("Img", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Img", width, height)

print("Camera Info:")
print("Resolution:", width, "x", height)
print("Frame Rate:", fps)

while True:
    ret, img=cap.read()

    img = cv2.flip(img, 1)

    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    hmin=cv2.getTrackbarPos("Hue Min", "HSV")
    hmax=cv2.getTrackbarPos("Hue Max", "HSV")
    smin=cv2.getTrackbarPos("Sat Min", "HSV")
    smax=cv2.getTrackbarPos("Sat Max", "HSV")
    vmin=cv2.getTrackbarPos("Val Min", "HSV")
    vmax=cv2.getTrackbarPos("Val Max", "HSV")

    if cv2.getTrackbarPos("H In Out","HSV")==0:
        hmin2=hmin
        hmax2=hmax
    else:
        hmin=0
        hmax=cv2.getTrackbarPos("Hue Min", "HSV")
        hmin2=cv2.getTrackbarPos("Hue Max", "HSV")
        hmax2=179
    if cv2.getTrackbarPos("S In Out","HSV")==0:
        smin2=smin
        smax2=smax
    else:
        smin=0
        smax=cv2.getTrackbarPos("Sat Min", "HSV")
        smin2=cv2.getTrackbarPos("Sat Max", "HSV")
        smax2=255
    if cv2.getTrackbarPos("V In Out", "HSV") == 0:
        vmin2 = vmin
        vmax2 = vmax
    else:
        vmin = 0
        vmax = cv2.getTrackbarPos("Val Min", "HSV")
        vmin2 = cv2.getTrackbarPos("Val Max", "HSV")
        vmax2 = 255

    lower=np.array([hmin,smin,vmin])
    upper=np.array([hmax,smax,vmax])
    lower2=np.array([hmin2,smin2,vmin2])
    upper2=np.array([hmax2,smax2,vmax2])
    mask=cv2.inRange(imgHSV,lower,upper)
    mask2=cv2.inRange(imgHSV,lower2,upper2)
    maskr=cv2.bitwise_or(mask,mask2)
    result=cv2.bitwise_and(img,img,mask=maskr)


    mask=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    mask2=cv2.cvtColor(mask2,cv2.COLOR_GRAY2BGR)
    maskr=cv2.cvtColor(maskr,cv2.COLOR_GRAY2BGR)
    #=============================================================================

    # Read an image
    # Read the original image
    original_image = maskr

    # Create a copy of the original image
    image_with_squares = original_image.copy()

    # Convert the image to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold or any other method to obtain a binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Minimum contour area threshold
    min_area_threshold = cv2.getTrackbarPos("min squar size","HSV")

    # List to store the coordinates of previously drawn rectangles
    prev_rectangles = []


    # Function to check if a rectangle is intersecting with any of the previously drawn rectangles
    def is_intersecting(rect):
        for prev_rect in prev_rectangles:
            if rect[0] < prev_rect[2] and rect[2] > prev_rect[0] and rect[1] < prev_rect[3] and rect[3] > prev_rect[1]:
                return True
        return False


    # Iterate through each contour
    for contour in contours:
        # Get the area of the contour
        area = cv2.contourArea(contour)

        # If the contour area is greater than the threshold, draw a rectangle
        if area > min_area_threshold:
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Check if the current rectangle is not intersecting with any of the previously drawn rectangles
            if not is_intersecting((x, y, x + w, y + h)):
                # Draw a rectangle around the contour on the image with squares
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Add the coordinates of the current rectangle to the list of previous rectangles
                prev_rectangles.append((x, y, x + w, y + h))


    #=============================================================================


    hmask=np.hstack([mask,mask2])
    hstack=np.hstack([img,result])
    stack=np.vstack([hstack,hmask])

    cv2.imshow('Img', stack)

    if cv2.waitKey(1)==ord('q'):
        break

    if cv2.waitKey(1)==ord('r'):
        cv2.setTrackbarPos("Hue Min", "HSV", 0)
        cv2.setTrackbarPos("Hue Max", "HSV", 179)
        cv2.setTrackbarPos("Hue2 Min", "HSV", 0)
        cv2.setTrackbarPos("Hue2 Max", "HSV", 179)
        cv2.setTrackbarPos("Sat Min", "HSV", 0)
        cv2.setTrackbarPos("Sat Max", "HSV", 255)
        cv2.setTrackbarPos("Val Min", "HSV", 0)
        cv2.setTrackbarPos("Val Max", "HSV", 255)

cap.release()
cv2.destroyAllWindows()