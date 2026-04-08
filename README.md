# AI Spam Classifier

A Python machine learning project that classifies SMS messages as **spam** or **ham** using **scikit-learn** and **TF-IDF vectorization**.

## Features

- Loads and cleans SMS message data
- Converts text into numerical features with TF-IDF
- Trains a Naive Bayes classifier
- Predicts whether messages are spam or ham
- Uses a custom spam probability threshold for improved detection

## Technologies Used

- Python
- pandas
- scikit-learn

## Project Structure

ai-spam-classifier/
- data/
  - spam.csv
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

## What I Learned

- How to prepare text data for machine learning
- How TF-IDF vectorization works
- How to train and use a Naive Bayes classifier
- How prediction probabilities can be used with a custom threshold
- How to structure a machine learning project for GitHub

## Future Improvements

- Add user input from the terminal
- Save and reload the trained model
- Try other classifiers such as Logistic Regression
- Build a simple web app version