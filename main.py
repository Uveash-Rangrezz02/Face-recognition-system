import dlib
import cv2
import os
import json
import numpy as np

# Load person data from JSON
with open("data.json", "r") as f:
    person_data = json.load(f)

# Initialize HOG face detector and face recognizer
hog_face_detector = dlib.get_frontal_face_detector()
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load known faces
known_face_encodings = []
known_face_metadata = []

for filename in os.listdir("images"):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join("images", filename)
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = hog_face_detector(gray)

        for face in faces:
            shape = shape_predictor(gray, face)
            face_descriptor = face_rec_model.compute_face_descriptor(gray, shape)
            known_face_encodings.append(np.array(face_descriptor))
            name_key = os.path.splitext(filename)[0]
            known_face_metadata.append(person_data.get(name_key, {}))
            break  # Only take first face per image

# Face comparison helper
def compare_faces(known_encodings, face_encoding):
    distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
    if len(distances) == 0:
        return None, 1.0
    best_index = np.argmin(distances)
    return best_index, distances[best_index]

# Recognize from webcam
def recognize_from_webcam():
    print("Opening webcam... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = hog_face_detector(rgb_frame)

        for face in faces:
            shape = shape_predictor(rgb_frame, face)
            face_descriptor = np.array(face_rec_model.compute_face_descriptor(rgb_frame, shape))
            best_match_index, distance = compare_faces(np.array(known_face_encodings), face_descriptor)

            name = "Unknown"
            metadata = {}
            if distance < 0.6:
                metadata = known_face_metadata[best_match_index]
                name = metadata.get("name", "Unknown")

            left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            info = f"{metadata.get('name', '')} | Roll: {metadata.get('roll', '')}"
            cv2.putText(frame, info, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Address: {metadata.get('address', '')}, Age: {metadata.get('age', '')}",
                        (left, bottom + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Face Recognition (HOG + dlib)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Recognize from image
def recognize_from_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = hog_face_detector(rgb_image)

    for face in faces:
        shape = shape_predictor(rgb_image, face)
        face_descriptor = np.array(face_rec_model.compute_face_descriptor(rgb_image, shape))
        best_match_index, distance = compare_faces(np.array(known_face_encodings), face_descriptor)

        name = "Unknown"
        metadata = {}
        if distance < 0.6:
            metadata = known_face_metadata[best_match_index]
            name = metadata.get("name", "Unknown")

        print(f"Detected: {metadata.get('name')} | Roll: {metadata.get('roll')} | Address: {metadata.get('address')} | Age: {metadata.get('age')}")

# Main
if __name__ == "__main__":
    mode = input("Choose mode:\n1 - Webcam\n2 - Image\n> ")
    if mode == "1":
        recognize_from_webcam()
    elif mode == "2":
        image_path = input("Enter image path: ")
        recognize_from_image(image_path)
    else:
        print("Invalid mode selected.")
