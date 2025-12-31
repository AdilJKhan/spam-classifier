import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# Load & Inspect Data
# ===============================
raw_mail_data = pd.read_csv('mail_data.csv')

mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

mail_data['Category'] = mail_data['Category'].map({
    'spam': 0,
    'ham': 1
})

if mail_data['Category'].isnull().any():
    raise ValueError("Label mapping failed. Check Category values.")

X = mail_data['Message']
Y = mail_data['Category']

print("Class distribution:")
print(Y.value_counts())

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# ===============================
# TF-IDF Vectorization
# ===============================
feature_extraction = TfidfVectorizer(
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2),
    stop_words='english',
    lowercase=True
)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

# ===============================
# Model Training
# ===============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_features, Y_train)

# ===============================
# Evaluation
# ===============================
train_preds = model.predict(X_train_features)
test_preds = model.predict(X_test_features)

print("\nClassification Report:")
print(classification_report(Y_test, test_preds))

print("Accuracy on training data:", accuracy_score(Y_train, train_preds))
print("Accuracy on testing data:", accuracy_score(Y_test, test_preds))

# ===============================
# Save trained model and vectorizer
# ===============================
with open("spam_classifier.pkl", "wb") as file:
    pickle.dump((feature_extraction, model), file)

# ===============================
# Prediction Function
# ===============================
label_map = {
    0: "Spam mail",
    1: "Ham mail"
}

def predict_email(text, vectorizer, model):
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    return label_map[prediction]

# Test prediction
print(predict_email("Yeah he got in at 2 and was v apologetic. n had fallen out and she was actin like spoilt child and he got caught up in that. Till 2! But we won't go there! Not doing too badly cheers. You?",feature_extraction,model))