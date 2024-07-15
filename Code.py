#commit test


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings




# Load the dataset
data = pd.read_excel("C:/Users/manoj kumar/Downloads/modified_dataset.xlsx")

# Separate features and target variable
X = data[['Age', 'Gender', 'OutdoorJob', 'OutdoorActivities', 'SmokingHabit',
          'Humidity', 'Pressure', 'Temperature', 'UVIndex', 'WindSpeed']]
y = data['ACTScore']  # Assuming 'ACTScore' is the target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Support Vector Machine (SVM) classifier
svm_clf = SVC(kernel='linear', random_state=42)
svm_clf.fit(X_train_scaled, y_train)

# Train Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train)

# Predictions
svm_pred = svm_clf.predict(X_test_scaled)
rf_pred = rf_clf.predict(X_test_scaled)

# Evaluate models
print("Support Vector Machine Classifier:")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print("Classification Report:")
print(classification_report(y_test, svm_pred))

print("\nRandom Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report:")

# Print classification report
print(classification_report(y_test, rf_pred))

