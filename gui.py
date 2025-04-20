import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import nltk

# Download necessary NLTK data
nltk.download('punkt')

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Preprocess and transform the text
def transform_text(text):
    from nltk.tokenize import word_tokenize
    text = ' '.join(word_tokenize(text))  # Tokenization
    return text

df['transformed_text'] = df['text'].apply(transform_text)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['transformed_text'], df['label'], test_size=0.2, random_state=42)

# Create a Naive Bayes model with CountVectorizer in a pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Function to predict the spam or ham
def predict_spam():
    message = text_area.get("1.0", "end-1c").strip()  # Get the message from the text area
    if message == "":
        messagebox.showerror("Error", "Please enter a message.")
    else:
        prediction = model.predict([message])[0]
        if prediction == "spam":
            result_label.config(text="This is a SPAM message.", fg="red")
        else:
            result_label.config(text="This is a HAM message.", fg="green")

# Create the main window
root = tk.Tk()
root.title("SMS Spam Classifier")
root.geometry("600x400")
root.config(bg='#f0f0f0')

# Create a heading label
heading_label = tk.Label(root, text="SMS Spam Classifier", font=('Arial', 24, 'bold'), bg='#f0f0f0', fg='#4CAF50')
heading_label.pack(pady=20)

# Create a text area widget for entering messages
text_area = tk.Text(root, height=6, width=50, font=('Arial', 14), bd=2, relief='sunken', wrap=tk.WORD, padx=10, pady=10)
text_area.pack(pady=10)

# Create a button to predict the message type
predict_button = tk.Button(root, text="Check Message", font=('Arial', 16), bg='#4CAF50', fg='white', relief='raised', command=predict_spam)
predict_button.pack(pady=20)

# Create a label to display the prediction result
result_label = tk.Label(root, text="", font=('Arial', 18), bg='#f0f0f0')
result_label.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
