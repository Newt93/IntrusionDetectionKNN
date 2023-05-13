from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset of incidents
incidents = [
    [1, 'Malware'],
    [2, 'Phishing'],
    [3, 'Malware'],
    [4, 'Unauthorized Access'],
    [5, 'Phishing'],
    [6, 'Unauthorized Access'],
]

# Split the dataset into features (X) and labels (y)
X = [incident[0] for incident in incidents]
y = [incident[1] for incident in incidents]

# Convert features to a 2D array (required by scikit-learn)
X = [[x] for x in X]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Classify a new incident
new_incident = [[7]]  # Provide the feature(s) of the new incident
predicted_label = knn.predict(new_incident)
print('Predicted label:', predicted_label)
