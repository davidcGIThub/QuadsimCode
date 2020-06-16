import cv2
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("InputFolder", help="Type in the folder name for input images")
args = parser.parse_args()
input_path  = os.path.dirname(os.path.realpath(__file__)) + "/" + args.InputFolder + "/"
output_path = os.path.dirname(os.path.realpath(__file__)) + "/" + "templates" + "/"
if not os.path.exists(output_path):
    os.mkdir(output_path)
template_num = 1

for filename in os.listdir(input_path):
    if (filename.endswith(".png")):
        #load the image 
        image_path = input_path + "/" + filename
        img_color = cv2.imread(image_path,3)

        #take grayscale of image
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) 

        #threshold the image
        ret, binary_map = cv2.threshold(img_gray, 160, 255, cv2.THRESH_BINARY)

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
            if cv2.contourArea(cnt) > maxsize:
                maxsize = cv2.contourArea(cnt)
                best = count
            count = count + 1
        x,y,w,h = cv2.boundingRect(contours[best])
        cropped_masked_color_image = masked_color_image[y:y+h,x:x+w]
        output_filename = "template" + str(template_num) + ".png"
        template_num += 1
        cv2.imwrite(output_path + output_filename, cropped_masked_color_image)