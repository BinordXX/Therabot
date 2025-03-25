import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("emotion_dataset_mapped.csv")
df['emotion'].value_counts().plot(kind='bar', title="Emotion Distribution")
plt.show()
