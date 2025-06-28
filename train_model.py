import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
data = pd.read_csv('data/heart_cleveland_upload.csv')

X = data.drop('condition', axis=1)
y = data['condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize learners
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100)
knn = KNeighborsClassifier(n_neighbors=5)

# Voting Classifier
ensemble = VotingClassifier(estimators=[
    ('lr', lr),
    ('rf', rf),
    ('knn', knn)
], voting='soft')

# Train model
ensemble.fit(X_train, y_train)

# Save model
with open('model/model.pkl', 'wb') as f:
    pickle.dump(ensemble, f)