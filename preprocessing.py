#--- amaanras.petersen@gmail.com ---
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer

nltk.download('stopwords')

#loading dataset 
df = pd.read_csv('Product review sentiment - Sheet1.csv')


# Displaying the initial dataframe
# print(df.head())


# Text processing  'Review Text' is column in csv
#keeping some punctuation marks
punctuation_to_keep = ['!', '?']
df['Review Text'] = df['Review Text'].apply(lambda x: ''.join([c for c in x if c.isalnum() or c.isspace() or c in punctuation_to_keep]))

df['Review Text'] = df['Review Text'].str.lower() # converting all letter to lower case to make is easier and cosistant

# tokenizer --using nltk.tokenize.WordPunctTokenizer rather
# df['Review Text'] = df['Review Text'].str.split()
tokenizer = WordPunctTokenizer()
df['Review Text'] = df['Review Text'].apply(tokenizer.tokenize)

# stopword removal....common words that dont carry any signaficant meaning
stop_words = set(stopwords.words('english'))
df['Review Text'] = df['Review Text'].apply(lambda x: [word for word in x if word not in stop_words])

# stemming avoiding having multiple versions of the same word
stemmer = PorterStemmer()
df['Review Text'] = df['Review Text'].apply(lambda x: [stemmer.stem(word) for word in x])

# displaying the preprocessed dataframe
#print(df.to_string(index=False)) 
print(df.head())

#saving the preprocessed dataframe to a csv file
df.to_csv('preprocessed_data.csv', index = False)
