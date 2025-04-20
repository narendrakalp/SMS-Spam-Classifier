## 	SMS Spam Classifier - Project Documentation


## 	1. Introduction
Spam messages are a significant concern for mobile users, often causing annoyance and security threats. In this project, we aim to build an SMS Spam Classifier using machine learning. The model will predict whether a given SMS message is spam or not. This is a classification problem where the objective is to categorize a message as either "spam" or "ham" (non-spam). The classifier uses machine learning techniques such as Natural Language Processing (NLP) to analyze text data.

This project leverages Naive Bayes, a well-known machine learning algorithm, combined with CountVectorizer to transform the text data into numerical features. The entire model is deployed using a graphical user interface (GUI) developed with Tkinter, allowing users to interact with the classifier easily by inputting SMS messages and receiving predictions.



## 	2. Problem Statement
The main objective of this project is to develop an SMS spam classifier using machine learning. The model should:

Take an input message (SMS) from the user.
Classify the message as either "spam" or "ham" (non-spam).
Provide real-time feedback to the user through a graphical interface.
By automating the identification of spam messages, users can filter unwanted content and reduce the risk of phishing attacks, fraudulent messages, and other types of spam.



## 	3. Project Requirements
The following software and libraries are required to run the SMS Spam Classifier:

## 	3.1 Software Requirements:
Python: The programming language used for this project. Python version 3.x is recommended.
Tkinter: A GUI toolkit used to build the interactive interface.
Jupyter Notebook (optional): For experimenting with machine learning models.
IDE/Text Editor: Any editor like Visual Studio Code, PyCharm, or Jupyter Notebook can be used for coding.

CSV File (spam.csv): A dataset containing SMS messages labeled as spam or ham.
## 	3.2 Libraries:
Pandas: For data manipulation and reading CSV files.
Scikit-learn: For implementing machine learning algorithms like Naive Bayes and CountVectorizer.
NLTK: Natural Language Toolkit, used for text processing (tokenization).
Matplotlib (optional): For data visualization (if needed).
NumPy: Used for numerical operations.

## 	3.3 Hardware Requirements:
A computer with internet access to download necessary libraries.
Minimal system resources required for the model training and GUI.



## 	4. Dataset
The dataset used in this project is the SMS Spam Collection Dataset available publicly on many data science platforms. It contains a collection of SMS messages that are pre-labeled as either "spam" or "ham." The dataset is used to train and test the model.

The dataset is organized as follows:

Column 1: v1 (Label): Contains the label for each message, either "spam" or "ham".
Column 2: v2 (Text): Contains the actual SMS message.
For example:

v1	v2
ham	Go until jurong point, crazy...
spam	Free entry in 2 a wkly comp...
The goal is to train the classifier to predict whether an unseen message belongs to the "spam" or "ham" category.



## 	5. Methodology
## 	5.1 Data Preprocessing
Before training the machine learning model, the text data requires preprocessing. The following steps were carried out:

Tokenization: The text is split into individual words (tokens) to better analyze the content of the message.
Vectorization: A CountVectorizer is used to convert the text messages into numerical data, which can be understood by the machine learning model. This step creates a bag of words from the messages, where each word in the vocabulary corresponds to a feature.
Data Splitting: The data is divided into training and testing sets. Typically, 80% of the data is used for training, and the remaining 20% is used for testing the model.

## 	5.2 Model Selection
For this classification task, Naive Bayes is used as the machine learning algorithm. Specifically, the Multinomial Naive Bayes classifier is employed, as it works well with discrete data such as word counts in text classification problems.

Why Naive Bayes?

Simplicity: It is easy to implement and computationally efficient.
Effectiveness: Works well with text classification tasks.
Assumptions: Assumes that each feature (word) is independent, which may not always hold true but still performs reasonably well in practice.

## 	5.3 Model Training
The model is trained using the preprocessed training data (X_train and y_train). The CountVectorizer converts the text into a format that the Naive Bayes classifier can understand. Once the model is trained, it can be used to predict new unseen SMS messages.

## 	5.4 Model Evaluation
After training the model, it is tested on unseen data (X_test and y_test). The performance is evaluated based on its accuracy, precision, recall, and F1 score. In this project, accuracy is the primary evaluation metric, but other metrics can be considered depending on the application.



## 	6. System Architecture
## 	6.1 Overview
The system consists of two main components:

Machine Learning Model: The core of the system that classifies SMS messages as spam or ham.
Graphical User Interface (GUI): The front-end interface where the user inputs the message and receives the prediction.

## 	6.2 Components
CountVectorizer:

Used to convert text messages into numerical features.
Each word is represented as a feature in a high-dimensional vector space.
Multinomial Naive Bayes:

The classification algorithm used to train the model.
Assumes that the presence of a word in a message is independent of other words.
Tkinter GUI:

Provides a simple interface for users to input SMS messages and receive spam/ham predictions.
The interface has an input box, a button to trigger classification, and a label to display the result.


## 	7. Implementation
## 	7.1 Tkinter GUI Setup
The graphical interface is developed using Tkinter, which is a Python library for creating desktop applications. The interface includes:

A Heading Label that displays "SMS Spam Classifier".
A Text Area for entering the SMS message.
A Button that triggers the classification when clicked.
A Result Label that shows whether the message is spam or ham.

## 	7.2 Code Explanation
Preprocessing the Data

def transform_text(text):
    from nltk.tokenize import word_tokenize
    text = ' '.join(word_tokenize(text))  # Tokenization
    return text
This function takes an input SMS message and tokenizes it (splits it into individual words).

Creating the Model

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
A pipeline is created where the CountVectorizer is used to convert the SMS messages into numerical features, and the Multinomial Naive Bayes classifier is used to train the model.

Prediction Function

def predict_spam():
    message = text_area.get("1.0", "end-1c").strip()
    if message == "":
        messagebox.showerror("Error", "Please enter a message.")
    else:
        prediction = model.predict([message])[0]
        if prediction == "spam":
            result_label.config(text="This is a SPAM message.", fg="red")
        else:
            result_label.config(text="This is a HAM message.", fg="green")
This function retrieves the input message, predicts whether it is spam or ham, and displays the result.

## 	7.3 Running the Application
When the user clicks the "Check Message" button, the classifier predicts whether the entered message is spam or ham and displays the result in a label with appropriate styling (green for ham and red for spam).



## 	8. Features
Spam Message Detection: The model can accurately classify messages as spam or ham.
Interactive GUI: The Tkinter-based interface allows for easy user interaction.
Real-time Prediction: Users can input messages and receive immediate feedback about whether the message is spam or ham.


## 	9. Future Enhancements
Support for Multiple Languages: Currently, the system works for English text. It can be extended to support other languages.
Improved Accuracy: The model's accuracy can be improved by experimenting with other machine learning algorithms like Support Vector Machines (SVM), Random Forest, or deep learning models.
Add Feature for Bulk SMS: Allow users to upload a file containing multiple messages for batch classification.


## 	10. Conclusion
This project provides a simple yet effective SMS Spam Classifier that can help users avoid spam messages. It integrates machine learning with a user-friendly interface, making it accessible for non-technical users. The system is flexible and can be enhanced further to improve accuracy, support multiple languages, and offer additional features like bulk SMS classification.
