from flask import Flask,request,jsonify
from flask_cors import CORS
import pickle
import re

app = Flask(__name__)
CORS(app)

model = pickle.load(open("../model/fake_news_model.pkl","rb"))
vectorizer = pickle.load(open("../model/vectorizer.pkl","rb"))

fake_indicators=[
"shocking","breaking","secret","conspiracy",
"exposed","banned","unbelievable","scandal",
"miracle","cure"
]

@app.route("/predict",methods=["POST"])
def predict():

    news=request.json["news"]

    vector=vectorizer.transform([news])

    prediction=model.predict(vector)[0]

    probs=model.predict_proba(vector)[0]

    fake_prob=round(probs[0]*100,2)
    real_prob=round(probs[1]*100,2)

    result="REAL" if prediction==1 else "FAKE"

    keywords=[]

    for word in fake_indicators:
        if re.search(r"\b"+word+r"\b",news.lower()):
            keywords.append(word)

    explanation = (
        "Suspicious keywords detected: "+", ".join(keywords)
        if keywords else
        "No suspicious linguistic patterns detected."
    )

    return jsonify({
        "prediction":result,
        "fake_probability":fake_prob,
        "real_probability":real_prob,
        "explanation":explanation
    })

if __name__=="__main__":
    app.run(debug=True)