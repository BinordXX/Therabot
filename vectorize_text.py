import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load preprocessed data
df = pd.read_csv("emotion_dataset.csv")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save TF-IDF vectors
joblib.dump(X_train_tfidf, "X_train_tfidf.pkl")
joblib.dump(X_test_tfidf, "X_test_tfidf.pkl")

# Save raw labels separately (for training script)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Save the vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Text vectorization complete. Your words now speak numbers.")
