from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import load_intents, clean_text
import random

print("🤖 Chatbot: Hello! Type 'quit' to exit.")

corpus, tags, responses = load_intents()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([clean_text(q) for q in corpus])

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("🤖 Chatbot: Bye!")
        break

    user_vec = vectorizer.transform([clean_text(user_input)])
    similarity = cosine_similarity(user_vec, X)
    max_sim = similarity.max()

    if max_sim > 0.3:
        index = similarity.argmax()
        tag = tags[index]
        print("🤖 Chatbot:", random.choice(responses[tag]))
    else:
        print("🤖 Chatbot: Sorry, I didn't understand that.")
