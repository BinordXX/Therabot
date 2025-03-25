import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load vectorized data
X_train_tfidf = joblib.load("X_train_tfidf.pkl")
X_test_tfidf = joblib.load("X_test_tfidf.pkl")

# Load raw string labels
y_train = pd.read_csv("y_train.csv")["label"]
y_test = pd.read_csv("y_test.csv")["label"]

# Fit encoder on STRING labels
label_encoder = LabelEncoder()
label_encoder.fit(y_train.tolist() + y_test.tolist())  # Fit on full label set
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train_encoded)

# Predict
y_pred = model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test_encoded, y_pred)
print("🎯 Accuracy:", accuracy)

# Classification report with correct string labels
target_names = label_encoder.classes_.astype(str)  # ensure they are strings
print("\n📝 Classification Report:\n", classification_report(y_test_encoded, y_pred, target_names=target_names))
