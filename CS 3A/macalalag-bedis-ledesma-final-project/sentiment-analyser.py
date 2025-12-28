import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("sentiment_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


def predict_review(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200)
    pred = model.predict(padded)[0][0]

    if pred > 0.5:
        return "positive", float(pred)
    else:
        return "negative", float(pred)


while True:
    review = input("\nEnter a review (or type 'quit' to exit):\n")
    if review.lower() == "quit":
        break

    sentiment, confidence = predict_review(review)
    print(f"Sentiment: {sentiment}  |  Confidence: {confidence:.2f}")
