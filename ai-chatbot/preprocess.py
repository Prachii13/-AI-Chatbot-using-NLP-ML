import json
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

def load_intents():
    with open("intents.json") as file:
        data = json.load(file)
    corpus, tags, responses = [], [], {}
    for intent in data['intents']:
        for pattern in intent['patterns']:
            corpus.append(pattern)
            tags.append(intent['tag'])
        responses[intent['tag']] = intent['responses']
    return corpus, tags, responses

def clean_text(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return ' '.join([w for w in tokens if w not in stop_words and w not in string.punctuation])
