import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from colorama import Fore, Style





df = pd.read_csv('preprocessed_data.csv')

#splitting the dataset into features and lables
x = df['Review Text'] # Features (review text)
y = df['Sentiment'] # Labels (sentiment)

# for extraction using TF-IDF Vectorization because of relativley small dataset 

# Initialize the vectorizer
vectorizer = TfidfVectorizer()

# Fit-transform the preprocessed text data
X_tfidf = vectorizer.fit_transform(x)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"{Fore.BLUE}{Style.BRIGHT}Sentiment Analysis Model Performance")
print("-----------------------------")
print(f"{Fore.GREEN}Accuracy: {accuracy}")
print(f"{Fore.YELLOW}Precision: {precision}")
print(f"{Fore.MAGENTA}Recall: {recall}")
print(f"{Fore.CYAN}F1-score: {f1}")
print(Style.RESET_ALL)