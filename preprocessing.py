import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

#loading dataset 
df = pd.read_csv('Product review sentiment - Sheet1.csv')

# Displaying the initial dataframe
print(df.head())

# Text processing  'Review Text' is column in csv
df['Review Text'] = df['Review Text'].str.replace('[^\w\s]','') # removing the punctions
df['Review Text'] = df['Review Text'].str.lower() # converting all letter to lower case to make is easier and cosistant
df['Review Text'] = df['Review Text'].str.split() # tokenization

