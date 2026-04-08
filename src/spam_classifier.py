import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

SPAM_THRESHOLD = 0.40


def load_data():
    df = pd.read_csv("data/spam.csv", sep="\t", header=None, names=["label", "message"])
    df["message"] = df["message"].str.strip()
    df = df.dropna()
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def train_model(df):
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

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    return model, vectorizer


def predict_message(model, vectorizer, message):
    message_vec = vectorizer.transform([message])
    probabilities = model.predict_proba(message_vec)[0]

    spam_prob = probabilities[1]
    label = "SPAM" if spam_prob >= SPAM_THRESHOLD else "HAM"

    return label, probabilities


def main():
    df = load_data()
    model, vectorizer = train_model(df)

    test_messages = [
        "Congratulations! You've won a free iPhone, click here now!",
        "Hey, are we still meeting tomorrow?",
        "URGENT! Claim your prize now!!!",
        "Can you send me the notes from class?"
    ]

    print("\nCustom Predictions:")
    for msg in test_messages:
        label, probs = predict_message(model, vectorizer, msg)
        print(f"\nMessage: {msg}")
        print("Prediction:", label)
        print(f"Spam probability: {probs[1]:.4f}")


if __name__ == "__main__":
    main()