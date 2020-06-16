#template matching
import cv2
import numpy as np
import os
import argparse
import imutils

#parser = argparse.ArgumentParser()
#parser.add_argument("InputFolder", help="Type in the folder name for input images")
#args = parser.parse_args()
#input_path  = os.path.dirname(os.path.realpath(__file__)) + "/" + args.InputFolder + "/"
input_path  = os.path.dirname(os.path.realpath(__file__)) + "/templates/"

#load scene
scene_path = os.path.dirname(os.path.realpath(__file__)) + "/" + "scene.png" 
image_scene = cv2.imread(scene_path,3)
(hs, ws, scene_channels) = image_scene.shape

num_sizes = 10
rotation_step_size = 30
rotation = 90
closest_template = np.zeros((1,1,3))

found = (None)
for filename in os.listdir(input_path):
    if (filename.endswith(".png")):
        template_path = input_path + "/" + filename
        template_color = cv2.imread(template_path,3)
        (ht, wt, temp_channels) = template_color.shape
        template_diag = np.sqrt(ht*ht + wt*wt)

        #calcute the max ratio of template size
        R = min([hs , ws])/template_diag
        found = (None)

        for scale in np.linspace(1.0*R/num_sizes, R, num_sizes)[::-1]:
            # resize the template according to the scale, and keep track of the ratio of the resizing
            resized_template = imutils.resize(template_color, width = int(template_color.shape[1] * scale))
            for angle in np.arange(0, rotation, rotation_step_size):
                rotated_template = imutils.rotate_bound(resized_template, angle)
                (h, w, temp_channels) = rotated_template.shape
                rotated_gray_template = cv2.cvtColor(rotated_template, cv2.COLOR_BGR2GRAY)
                img, contours, hierarchy = cv2.findContours(rotated_gray_template,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                x,y,w,h = cv2.boundingRect(contours[0])
                cropped_rotated_template = rotated_template[y:y+h,x:x+w]
                result = cv2.matchTemplate(image_scene, cropped_rotated_template, cv2.TM_CCORR_NORMED)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
                # if we have found a new maximum correlation value, then update the bookkeeping variable
                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, h, w)
                    closest_template = cropped_rotated_template
                (startX, startY) = (maxLoc[0] , maxLoc[1])
                (endX, endY) = int(maxLoc[0] + w ), int(maxLoc[1] + h)
                image_scene_copy = image_scene.copy()
                cv2.rectangle(image_scene_copy, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.imshow("Estimated_Match", image_scene_copy)
                cv2.imshow("Template", cropped_rotated_template)
                cv2.waitKey(100)
            
        # unpack the bookkeeping variable and compute the (x, y) coordinates of the bounding box
cv2.destroyWindow("Template")
cv2.destroyWindow("Estimated_Match")
(maxVal , maxLoc, h , w) = found
(startX, startY) = (maxLoc[0] , maxLoc[1])
(endX, endY) = int(maxLoc[0] + w), int(maxLoc[1] + h)
# draw a bounding box around the detected result and display the image
cv2.rectangle(image_scene, (startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.imshow("Best_Match", image_scene)
cv2.imshow("Closest_Template", closest_template)
cv2.waitKey(0)