import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler

# === Load the vectorized training data ===
X_train = joblib.load("X_train_tfidf.pkl")  # Corrected filename
y_train = pd.read_csv("y_train.csv").squeeze()  # Ensure it's a Series

# === Oversample minority classes ===
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

print("✅ Oversampling complete. New class distribution:")
print(pd.Series(y_resampled).value_counts())

# === Train the Logistic Regression model ===
model = LogisticRegression(max_iter=1000)
model.fit(X_resampled, y_resampled)

# === Save the trained model ===
joblib.dump(model, "emotion_model.pkl")

# === Optional: Evaluate on training data ===
y_train_pred = model.predict(X_resampled)
print("\n📈 Training Accuracy:", accuracy_score(y_resampled, y_train_pred))
print("\n📋 Classification Report:\n", classification_report(y_resampled, y_train_pred))

print("✅ Model training complete. Saved as emotion_model.pkl")
