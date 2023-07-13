import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv('preprocessed_data.csv')

#splitting the dataset into features and lables
x = df['Review Text'] # Features (review text)
y = df['Sentiment'] # Labels (sentiment)

# for extraction using TF-IDF Vectorization because of relativley small dataset 

# Initialize the vectorizer
vectorizer = TfidfVectorizer()

# Fit-transform the preprocessed text data
X_tfidf = vectorizer.fit_transform(x)