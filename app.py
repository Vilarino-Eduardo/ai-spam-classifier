import joblib
from flask import Flask, render_template, request

SPAM_THRESHOLD = 0.40
MODEL_PATH = "model/spam_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_message(message):
    message_vec = vectorizer.transform([message])
    probabilities = model.predict_proba(message_vec)[0]

    spam_prob = probabilities[1]
    label = "SPAM" if spam_prob >= SPAM_THRESHOLD else "HAM"

    return label, spam_prob


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        message = request.form["message"]
        prediction, probability = predict_message(message)

        return render_template(
            "index.html",
            prediction=prediction,
            probability=f"{probability:.4f}",
            message=message
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)