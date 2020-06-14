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
binary_filtered_map_color = cv2.cvtColor(filtered_binary_map, cv2.COLOR_GRAY2BGR)
masked_color_image = cv2.bitwise_and(img_color, binary_filtered_map_color)
masked_gray_image = cv2.bitwise_and(img_gray,filtered_binary_map)

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
cropped_masked_color_image = masked_color_image[y:y+h,x:x+w]
cropped_gray_image = img_gray[y:y+h,x:x+w]
cropped_color_image = img_color[y:y+h,x:x+w]
cropped_filtered_binary_map = filtered_binary_map[y:y+h,x:x+w]
#template = cv2.Canny(cropped_masked_gray_image,100,255)
template = cropped_color_image
(ht, wt) = cropped_gray_image.shape[::-1]
mask = cropped_filtered_binary_map

#load scene to check
path_image_scene = "/home/david/Videos/quad_unwashed/out32.png"
image_scene = cv2.imread(path_image_scene,3)
image_scene_gray = cv2.cvtColor(image_scene, cv2.COLOR_BGR2GRAY) 
#scene = cv2.Canny(image_scene_gray,200,250)
scene = image_scene

(hs, ws) = image_scene_gray.shape[::-1]

R = min([hs/ht , ws/wt])
found = (None)
cv2.imshow("template", template)
cv2.waitKey(0)
cv2.imshow("scene", scene)
cv2.waitKey(0)
num_sizes = 20

for scale in np.linspace(1.0*R/num_sizes, R, num_sizes)[::-1]:
    # resize the template according to the scale, and keep track
    # of the ratio of the resizing
    resized_template = imutils.resize(template, width = int(template.shape[1] * scale))
    result = cv2.matchTemplate(scene, resized_template, cv2.TM_CCORR_NORMED, mask = mask)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    # if we have found a new maximum correlation value, then update the bookkeeping variable
    if found is None or maxVal > found[0]:
        found = (maxVal, maxLoc, scale)
    
# unpack the bookkeeping variable and compute the (x, y) coordinates of the bounding box
(maxVal , maxLoc, r) = found
(startX, startY) = (maxLoc[0] , maxLoc[1])
(endX, endY) = int(maxLoc[0] + wt * r), int(maxLoc[1] + ht * r)
# draw a bounding box around the detected result and display the image
cv2.rectangle(image_scene, (startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.imshow("Image", image_scene)
cv2.waitKey(0)