import cv2
import os

def detect_faces_batch(input_folder, output_folder):

    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if not os.path.exists(input_folder):
        print(f"Error: The folder '{input_folder}' does not exist.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Starting processing on '{input_folder}'...")
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No image files found.")
        return

    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
        faces_profile = profile_cascade.detectMultiScale(gray, 1.05, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for (x, y, w, h) in faces_profile:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)
        
    print("Faces detected")



input_dir = "input_image"
output_dir = "output_image"
    
detect_faces_batch(input_dir, output_dir)