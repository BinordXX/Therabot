import joblib

# Load model, vectorizer, and label-to-emotion mapping
model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_to_emotion = joblib.load("label_to_emotion.pkl")

print("🧠 Therabot is listening. Speak your mind.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Therabot 💤: Goodbye, take care of your mind. 🌙")
        break

    # Vectorize the input text
    input_vector = vectorizer.transform([user_input])

    # Predict emotion (numeric label)
    predicted_label = model.predict(input_vector)[0]

    # Map numeric label to emotion
    predicted_emotion = label_to_emotion.get(predicted_label, "uncertain")

    print(f"Therabot 🪄: It sounds like you're feeling *{predicted_emotion}*.\n")
