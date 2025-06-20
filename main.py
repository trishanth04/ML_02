#  Sentiment Analysis with TF-IDF and Logistic Regression

#  Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#  Sample Dataset: Simulated Customer Reviews
data = {
    'review': [
        "I love this product! It's amazing.",
        "Terrible experience. Will not buy again.",
        "Great quality and fast delivery!",
        "Worst customer service ever.",
        "Highly recommended to everyone.",
        "Completely useless, very disappointed.",
        "This was exactly what I needed.",
        "Not worth the money.",
        "Fantastic! Exceeded my expectations.",
        "Very bad. Do not purchase."
    ],
    'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
}

#  Convert to DataFrame
df = pd.DataFrame(data)

# Preprocessing

#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.3, random_state=42)

#  TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#  Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

#  Predict
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Accuracy: {accuracy * 100:.2f}%")

#  Classification Report
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

#  Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Sentiment Analysis")
plt.tight_layout()
plt.savefig("sentiment_confusion_matrix.png")
plt.show()
