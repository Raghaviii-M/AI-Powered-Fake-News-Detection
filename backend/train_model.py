import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("../dataset/fake_or_real_news.csv")

# Remove unnecessary column
if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])

# Combine title and text
data["content"] = data["title"] + " " + data["text"]

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]','',text)
    return text

data["content"] = data["content"].apply(clean_text)

# Convert labels
data["label"] = data["label"].map({
    "REAL":1,
    "FAKE":0
})

X = data["content"]
y = data["label"]

# Train/test split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# TF-IDF with bigrams
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.8,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Balanced logistic regression
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_vec,y_train)

pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test,pred)

print("Model Accuracy:",accuracy)

# Save model
pickle.dump(model,open("../model/fake_news_model.pkl","wb"))
pickle.dump(vectorizer,open("../model/vectorizer.pkl","wb"))

print("Model saved successfully")