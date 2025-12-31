import os
import pickle

# ==================================================
# Load the saved TF-IDF vectorizer and trained model
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_classifier.pkl")

with open(MODEL_PATH, "rb") as file:
    feature_extraction, model = pickle.load(file)

# Map numeric model outputs to human-readable labels
label_map = {
    0: "Spam mail",
    1: "Ham mail"
}

# ===============================
# Prediction function
# ===============================
def predict_email(text):
    features = feature_extraction.transform([text])
    prediction = model.predict(features)[0]
    return label_map[prediction]

# Test
print(predict_email("PRIVATE! Your 2004 Account Statement for 07742676969 shows 786 unredeemed Bonus Points. To claim call 08719180248 Identifier Code: 45239 Expires"))
print(predict_email("Thanks a lot for your wishes on my birthday. Thanks you for making my birthday truly memorable."))

# ===============================
# CLI Interface
# ===============================
print("\nSpam Detector CLI")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("> ")

    if user_input.lower() == "exit":
        print("Exiting spam detector.")
        break

    result = predict_email(user_input)
    print("Prediction:", result, "\n")
