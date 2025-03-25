import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Load the CSV file
df = pd.read_csv("emotion_dataset.csv")  # Replace with your actual filename

# Basic text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters & digits
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Optional: shuffle the dataset to avoid bias
df = df.sample(frac=1).reset_index(drop=True)

# Split into training and test sets
X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the preprocessed sets (optional, for reuse)
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("✅ Preprocessing complete. Data cleaned and split!")
