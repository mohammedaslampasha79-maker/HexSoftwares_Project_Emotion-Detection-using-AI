import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("emotion_data.csv")

X = data['text']
y = data['emotion']

# Convert text to numbers
vectorizer = CountVectorizer(lowercase=True, stop_words='english')
X_vectors = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectors, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# User input
while True:
    text = input("\nEnter text (or 'exit'): ")
    if text.lower() == "exit":
        break
    vector = vectorizer.transform([text])
    emotion = model.predict(vector)
    print("Detected Emotion:", emotion[0])
