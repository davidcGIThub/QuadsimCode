import cv2
import numpy as np
import imutils

#load the image 
path_image = "/home/david/Videos/quad_unwashed/out1.png"
img_color = cv2.imread(path_image,3)

#take grayscale of image
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) 

#threshold the image
ret, binary_map = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY) 

# Remove noise from the image, background dots
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
areas = stats[1:,cv2.CC_STAT_AREA]
filtered_binary_map = np.zeros((labels.shape), np.uint8)
for i in range(0, nlabels - 1):
    if areas[i] >= 200:   #keep
        filtered_binary_map[labels == i + 1] = 255

#produce image with background removed
# binary_filtered_map_color = cv2.cvtColor(filtered_binary_map, cv2.COLOR_GRAY2BGR)
# image_background_removed = cv2.bitwise_and(img_color, binary_filtered_map_color)

#get contour around objects in view
image, contours, hierarchy = cv2.findContours(filtered_binary_map,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# select the largest contour in the image and create and crop to create template
best = 0
maxsize = 0
count = 0
for cnt in contours:
    if cv2.contourArea(cnt) > maxsize :
        maxsize = cv2.contourArea(cnt)
        best = count
    count = count + 1

x,y,w,h = cv2.boundingRect(contours[best])
template = filtered_binary_map[y:y+h,x:x+w]
(tH, tW) = template.shape[::-1]

#load scene to check
path_image = "/home/david/Videos/quad_unwashed/out1.png"
image_scene = cv2.imread(path_image,3)

# # Apply template Matching
# w, h = template.shape[::-1]
# res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# top_left = max_loc
# bottom_right = (top_left[0] + w, top_left[1] + h)
# cv2.rectangle(image_scene,top_left, bottom_right, 255, 2)
# cv2.imshow('image', image_scene)
# cv2.waitKey(0) 

found = None
for scale in np.linspace(0.1, 1.0, 20)[::-1]:
    # resize the image according to the scale, and keep track
    # of the ratio of the resizing
    resized = imutils.resize(img_gray, width = int(img_gray.shape[1] * scale))
    r = img_gray.shape[1] / float(resized.shape[1])
    # if the resized image is smaller than the template, then break
    # from the loop
    if resized.shape[0] < tH or resized.shape[1] < tW:
        break
    # detect edges in the resized, grayscale image and apply template
    # matching to find the template in the image
    #edged = cv2.Canny(resized, 50, 200)
    result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    # if we have found a new maximum correlation value, then update
    # the bookkeeping variable
    if found is None or maxVal > found[0]:
        found = (maxVal, maxLoc, r)
# unpack the bookkeeping variable and compute the (x, y) coordinates
# of the bounding box based on the resized ratio
(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
# draw a bounding box around the detected result and display the image
cv2.rectangle(img_color, (startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.imshow("Image", img_color)
cv2.waitKey(0)