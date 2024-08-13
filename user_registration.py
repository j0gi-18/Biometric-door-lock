import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import os

# Load the pre-trained model
model_path = '/Users/sagarkumbhar/Documents/TLC_Polymers_Ltd./New Algo/MobileFaceNet_9925_9680.pb'
model = tf.compat.v1.GraphDef()
with tf.io.gfile.GFile(model_path, 'rb') as f:
    serialized = f.read()
    model.ParseFromString(serialized)
tf.import_graph_def(model, name='')

# function to generate face embeddings
def generate_embeddings(images):
    imgs = [cv2.resize(img, (112, 112)) for img in images]
    imgs = [img / 255.0 for img in imgs]
    imgs = np.stack(imgs, axis=0)

    with tf.compat.v1.Session() as sess:
        images_placeholder = sess.graph.get_tensor_by_name("input:0")
        embeddings = sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")

        feed_dict = {images_placeholder: imgs, phase_train_placeholder: False}
        embeddings = sess.run(embeddings, feed_dict=feed_dict)

    return embeddings

user_name = input("Please enter your name: ")

cap = cv2.VideoCapture(0)

# Create a Mediapipe Face Detection object
mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=2, min_detection_confidence=0.3)
mp_draw = mp.solutions.drawing_utils

# Image capture
captured_images = []
max_frames = 200 

while len(captured_images) < max_frames:
    ret, image = cap.read()
    if not ret:
        break

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = mp_face_detection.process(rgb_image)

    # Draw the face detection results on the image
    if results.detections:
        for detection in results.detections:
            mp_draw.draw_detection(image, detection)
            # Capture the image if a face is detected
            captured_images.append(image)

    cv2.imshow('Face Detection', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Exit
cap.release()
cv2.destroyAllWindows()

# Generate embeddings from the captured images
if captured_images:
    embeddings = generate_embeddings(captured_images)

    # Save the embeddings to a file using the user's name
    np.save(f'/Users/sagarkumbhar/Documents/TLC_Polymers_Ltd./new_user_embeddings/{user_name}_embeddings.npy', embeddings)
    print(f"Embeddings saved successfully as '{user_name}_embeddings.npy'.")

    # Update the text file with the user's name
    with open('/Users/sagarkumbhar/Documents/TLC_Polymers_Ltd./labels.txt', 'a') as f:
        f.write(user_name + '\n')  # Append the name followed by a newline

    print(f"User name '{user_name}' added to 'user_names.txt'.")
else:
    print("Error: No faces detected during the capture period.")