import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import joblib

# Load training data
df = pd.read_csv("D:/data_science/project-chatbot/Data/chatbot_training_data.csv")
x = df['text'].values
y = df['intent'].values

# Define pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english')),
    ('svc', SVC(probability=True))
])

# Train model using grid search
params = {
    'svc__C': [1],
    'svc__kernel': ['linear'],
    'svc__gamma': ['scale'],
    'svc__class_weight': ['balanced']
}

grid = GridSearchCV(pipeline, param_grid=params, cv=3)
grid.fit(x, y)

# Save the trained model (VERY IMPORTANT: use grid.best_estimator_)
joblib.dump(grid.best_estimator_, 'chatbot_model.pkl')
