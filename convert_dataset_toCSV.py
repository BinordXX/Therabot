from datasets import load_dataset
import pandas as pd

# Load the full unsplit version of the dataset
dataset = load_dataset("dair-ai/emotion", "unsplit")

# Convert to a pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Save to CSV

label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
df['emotion'] = df['label'].map(label_map)
df.to_csv("emotion_dataset_mapped.csv", index=False)


print("✅ emotion_dataset.csv saved successfully!")
