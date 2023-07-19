import pandas as pd
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
import pickle

nltk.download('wordnet')

# Loading dataset
df = pd.read_csv('Product review sentiment - Sheet1.csv')

# Text processing - 'Review Text' is a column in the CSV

# Keeping some punctuation marks
punctuation_to_keep = ['!', '?']
df['Review Text'] = df['Review Text'].apply(lambda x: ''.join([c for c in x if c.isalnum() or c.isspace() or c in punctuation_to_keep]))
df['Review Text'] = df['Review Text'].str.lower()  # Converting all letters to lowercase to make it easier and consistent

# Tokenization using nltk.tokenize.WordPunctTokenizer
tokenizer = WordPunctTokenizer()
df['Review Text'] = df['Review Text'].apply(tokenizer.tokenize)

# Lemmatization
lemmatizer = WordNetLemmatizer()
df['Review Text'] = df['Review Text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Displaying the preprocessed dataframe
print(df.head())

# Saving the preprocessed DataFrame to a Pickle file
with open('preprocessed_data.pickle', 'wb') as f:
    pickle.dump(df, f)
