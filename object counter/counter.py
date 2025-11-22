import cv2
import numpy as np
import os

def count_red_objects(image_path):
    if not os.path.exists(image_path):
        print("Error: Image not found.")
        return

    img = cv2.imread(image_path)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255]) 
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    mask = mask1 + mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    output_img = img.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            count += 1
            
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.putText(output_img, f"Red Object {count}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    print(f"Final Count: {count}")

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((output_img, mask_bgr))
    
    cv2.imshow("Result", stacked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

count_red_objects("q.jpg")