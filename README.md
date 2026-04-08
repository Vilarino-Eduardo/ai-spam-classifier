# AI Spam Classifier

A Python machine learning project that classifies SMS messages as **spam** or **ham** using **scikit-learn** and **TF-IDF vectorization**.

## Key Features

- Clean and preprocess raw text data
- Convert text into numerical vectors using TF-IDF
- Train a Multinomial Naive Bayes classifier
- Predict spam vs ham messages
- Custom probability threshold for improved spam detection
- Allows live message classification through terminal input
- Saves and reloads the trained model and vectorizer with joblib

## Technologies Used

- Python
- pandas
- scikit-learn

## Project Structure

ai-spam-classifier/
- data/
  - spam.csv
- model/
  - spam_model.pkl
  - vectorizer.pkl
- src/
  - spam_classifier.py
- requirements.txt
- README.md
- .gitignore

## How to Run

1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   cd ai-spam-classifier
   ```

2. Create and activate a virtual environment:

   - Windows

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the project:

   ```bash
   python src/spam_classifier.py
   ```

## Example Predictions

- "Congratulations! You've won a free iPhone, click here now!" → SPAM
- "Hey, are we still meeting tomorrow?" → HAM
- "URGENT! Claim your prize now!!!" → SPAM
- "Can you send me the notes from class?" → HAM

## CLI Example

```bash
AI Spam Classifier
Type a message to classify it.
Type 'exit' to quit.

Enter a message: Congratulations! You've won a free iPhone, click here now!
Prediction: SPAM
Spam probability: 0.4733

Enter a message: Hey, are we still meeting tomorrow?
Prediction: HAM
Spam probability: 0.0050

Enter a message: exit
Goodbye.
```

## What I Learned

- How to prepare text data for machine learning
- How TF-IDF vectorization works
- How to train and use a Naive Bayes classifier
- How prediction probabilities can be used with a custom threshold
- How to structure a machine learning project for GitHub

## Two models were tested for this project:

- Multinomial Naive Bayes
- Logistic Regression

# Results

    |        Model        | Accuracy | Spam Recall |
    |---------------------|----------|-------------|
    | Naive Bayes         | ~97.3%   | 0.80        |
    | Logistic Regression | ~97.1%   | 0.79        |

# Conclusion

Multinomial Naive Bayes performed slightly better on this dataset, especially in detecting spam messages.  
For this reason, it was chosen as the final model.

This comparison highlights the importance of testing multiple models rather than assuming a more complex model will perform better.

## Future Improvements

- Add user input from the terminal
- Save and reload the trained model
- Try other classifiers such as Logistic Regression
- Build a simple web app version