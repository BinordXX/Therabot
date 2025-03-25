import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("emotion_dataset_mapped.csv")

# Ensure the columns exist
assert all(col in df.columns for col in ["text", "label", "emotion"]), "Required columns not found!"

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save vectorized data
joblib.dump(X_train_tfidf, "X_train_tfidf.pkl")
joblib.dump(X_test_tfidf, "X_test_tfidf.pkl")

# Save labels
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Save vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Save label-to-emotion mapping (for decoding predictions)
label_to_emotion = dict(df[["label", "emotion"]].drop_duplicates().values.tolist())
joblib.dump(label_to_emotion, "label_to_emotion.pkl")

print("✅ Text vectorization complete. Your words now speak in vectors.")
print("✅ TF-IDF saved, labels saved, and emotion mapping preserved.")
