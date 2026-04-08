import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

SPAM_THRESHOLD = 0.40
MODEL_PATH = "model/spam_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"


def load_data():
    df = pd.read_csv("data/spam.csv", sep="\t", header=None, names=["label", "message"])
    df["message"] = df["message"].str.strip()
    df = df.dropna()
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def train_and_save():
    df = load_data()

    X = df["message"]
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("model", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Model and vectorizer saved.")

    return model, vectorizer

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Loaded existing model.")
        return model, vectorizer
    else:
        print("No saved model found. Training new model...")
        return train_and_save()


def predict_message(model, vectorizer, message):
    message_vec = vectorizer.transform([message])
    probabilities = model.predict_proba(message_vec)[0]

    spam_prob = probabilities[1]
    label = "SPAM" if spam_prob >= SPAM_THRESHOLD else "HAM"

    return label, probabilities


def main():
    model, vectorizer = load_model()

    print("\nAI Spam Classifier")
    print("Type a message to classify it.")
    print("Type 'exit' to quit.")

    while True:
        user_message = input("\nEnter a message: ")

        if user_message.lower() == "exit":
            print("Goodbye.")
            break

        label, probs = predict_message(model, vectorizer, user_message)

        print("Prediction:", label)
        print(f"Spam probability: {probs[1]:.4f}")


if __name__ == "__main__":
    main()