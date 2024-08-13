import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Function to load embeddings from a specified directory
def load_embeddings_from_directory(directory):
    embeddings = []
    labels = []
    label_index = 0 

    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            # Load embeddings
            file_path = os.path.join(directory, filename)
            person_embeddings = np.load(file_path)
            embeddings.append(person_embeddings)
            # Create labels based on the order of files
            labels.extend([label_index] * len(person_embeddings))
            label_index += 1  # Increment label for the next person

    return np.concatenate(embeddings, axis=0), np.array(labels)

# embeddings directory
embeddings_dir = '/Users/sagarkumbhar/Documents/TLC_Polymers_Ltd./Embeddings'

# Load embeddings and labels
X, y = load_embeddings_from_directory(embeddings_dir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Calculate accuracy
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

# Results
#print("Accuracy:", accuracy)
print("Classification report:\n", report)
print("Confusion matrix:\n", matrix)

# Save the model
joblib.dump(clf, 'randomforest.joblib')