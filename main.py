import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
from sklearn.ensemble import RandomForestClassifier
import joblib
import datetime as dt
import pandas as pd
import paho.mqtt.client as mqtt
import time

class FaceRecognition:
    def __init__(self):
        # Load models and resources
        self.mtcnn = MTCNN(thresholds=[0.9, 0.9, 0.9])
        self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
        self.clf = joblib.load("/Users/sagarkumbhar/Documents/TLC_Polymers_Ltd./classifiers/randomForest/randomforestClassifier.joblib")
        self.label_to_name = {0: 'A39 Akash Khulpe',  1: 'A56 Anuj Gavhane', 2:'A50 Devang Edle', 3:'A51 Deepanshu Gadling', 
                              4:'A45 Gaurav Diwedi', 5:'A41 Parimal Kumar', 6:'A40 Parth Deshpande',7:'A46 Rutuja Doiphode',
                              8:'A47 Sagar Kumbhar'}
        self.threshold = 0.7  
        self.attendance_records = []
        self.predicted_names = []
        self.broker_address = "localhost"
        self.topic = "hello/world"
        self.client = mqtt.Client()
        

    def recognize_faces(self, frame):
        """ This method takes frames and performs face recognition by detecting faces through MTCNN.
        The detected face area from the frames is fed to FaceNet and embeddings are extracted.
        After sucessfull identification of people the name and entry time are stored in a pandas 
        dataframe ans saved as a CSV file"""
        
        current_time = dt.datetime.now().strftime('%I: %M: %S %p')
        boxes, _ = self.mtcnn.detect(frame) 
        
        # checks whether bounding boxes(faces) are present in the frames
        if boxes is not None: 
            faces = [] 
            face_boxes = [] 
            
            for box in boxes: 
                x1, y1, x2, y2 = box.astype(int) 
                face = frame[y1:y2, x1:x2] 
                
                if face.size != 0: 
                    face = cv2.resize(face, (160, 160)) 
                    face = torch.from_numpy(face).permute(2, 0, 1).float() 
                    face = (face - 127.5) / 128.0 
                    faces.append(face) 
                    face_boxes.append((x1, y1, x2, y2)) 
                    
            if len(faces) > 0:
                # Convert faces list to a batch tensor
                faces = torch.stack(faces)

                # Generate embeddings for all faces in the batch
                with torch.no_grad():
                    embeddings = self.facenet_model(faces)

                # Perform predictions on the embeddings
                predictions = self.clf.predict(embeddings.numpy())
                probabilities = self.clf.predict_proba(embeddings.numpy())

                # Iterate over the predictions and draw bounding boxes
                for prediction, probability, box in zip(predictions, probabilities, face_boxes):
                    if probability.max() >= self.threshold:
                        predicted_name = self.label_to_name[int(prediction)]
                        #print("Predicted name:", predicted_name)
                        current_date = dt.datetime.now().strftime('%d-%m-%y')
                        current_confidence = probability.max()
                
                        if not any(record['Name'] == predicted_name for record in self.attendance_records):
                            # Append attendance record to the list
                            self.attendance_records.append({'Name': predicted_name, 'Date': current_date, 'Time': current_time})
                
                        #uncomment this to see probability/confidence of each class
                        #print(probabilities)  
                
                        cv2.putText(frame, predicted_name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (36, 255, 12), 2)
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, 'unknown person', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (36, 255, 12), 2)
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                    # Print all predicted names together
                    self.predicted_names = [self.label_to_name[int(prediction)] if probability.max() >= self.threshold else 'unknown person' for prediction, probability in zip(predictions, probabilities)]
                    #print("Predicted names:", predicted_names)

                    attendance_df = pd.DataFrame(self.attendance_records)

                    # Save the DataFrame to a CSV file
                    attendance_df.to_csv('attendance_records.csv', index=False)

    
    def run(self):
        """ This method initializes the camera using OpenCv and invokes the recognize_faces(self, frame) 
        method in a while loop it runs continously until quit key(q) is pressed. The run() methold also 
        contains the logic for opening the door or keeps it closed based on the recognition status of the person.
        Contains a communication protocol to send names of the identified person to the broker."""
        
        # turn on camera 
        cap = cv2.VideoCapture(0)
        
        # connect the client to the broker
        self.client.connect(self.broker_address)

        # ideally keeps the door closed
        door_status = False

        while True:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.recognize_faces(frame)
            print(self.predicted_names)

            try:
                if "unknown person" in self.predicted_names:
                    door_status = False  # closes the door
                    message = "Person unrecognised, Entry denied"
                    self.client.publish(self.topic, message)
                    
                else:
                    door_status = True  # door opened
                    message = "Welcome! to Tlc Polymers Ltd."
                    self.client.publish(self.topic, message)

            except KeyboardInterrupt:
                print("Exitining")
            
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        self.client.disconnect()
        cv2.destroyAllWindows()
    
# create an instance of the class 
face_recognition = FaceRecognition()
face_recognition.run()
