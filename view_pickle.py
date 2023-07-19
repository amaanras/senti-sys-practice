import pickle

# Load the preprocessed DataFrame from the Pickle file
with open('preprocessed_data.pickle', 'rb') as f:
    df = pickle.load(f)

# Now you can use the 'df' DataFrame as your preprocessed data
print(df)
