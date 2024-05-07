import os
import json
import pandas as pd
import numpy as np
import requests
import io
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

JSON_URL = 'https://gist.githubusercontent.com/Arshad-ashuu/ec5672a56f13c26216bcf3426fbf0a8b/raw/5339643ed5527678f8a87ee8cc22ddb4b95d8431/gistfile1.txt'
#Read the JSON File

try:
# Download the JSON content from the gist repository URL
    response = requests.get(JSON_URL)
    response.raise_for_status()  # Raise an exception for bad response status
    dt = pd.read_json(io.StringIO(response.text))
except requests.exceptions.RequestException as e:
    # Handle any exceptions that may occur during the download
    print(f"Error downloading JSON file: {e}")


X, Y = dt.code, dt.language 
#X holds codes and Y holds the programming language

x_train , x_test , y_train , y_test = train_test_split(X , Y, test_size=0.2) 
#Splitting data to train and test in a 80-20 ratio [changable by modifying test_size]

pattern = r"""\b[A-Za-z_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]`"']"""
#Creating a tokenization pattern using regex

vectorizer = TfidfVectorizer(token_pattern = pattern)
#Creating a TfidVectorizer with the tokenization pattern

x_train_tf = vectorizer.fit_transform(x_train)
# #Transforming data with TfidVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
#Import required libraries
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1500,batch_size=64)   #Multilayer Perceptron
mnb = MultinomialNB(alpha=0.1)  #Multinomial Naive Bayes
rm = RandomForestClassifier(max_depth = None, random_state = 42, n_estimators=100)   #RandomForest

#Trial
param_grid_mnb = {'alpha': [0.1, 0.5, 1.0], 'fit_prior' : [True,False]}
grid_search_mnb = GridSearchCV(mnb, param_grid_mnb, cv=5, scoring='accuracy')

grid_search_mnb.fit(x_train_tf,y_train)
#Training the Multinomial Naive Bayes Classifier
best_params_mnb = grid_search_mnb.best_params_
#Getting best hyperparameters
best_mnb_model = grid_search_mnb.best_estimator_

MultinomialNB(alpha=0.1)

rm.fit(x_train_tf,y_train)
#Training the RandomForest Classifier

RandomForestClassifier(random_state=42)

mlp.fit(x_train_tf,y_train)
#Training the Multilayer Perceptron Classifier [Convolutional Neural Network]

MLPClassifier(batch_size=64, hidden_layer_sizes=(50,), max_iter=1500)

best_mnb_model.fit(x_train_tf,y_train)

MultinomialNB(alpha=0.1)

x_test_tf = vectorizer.transform(x_test)
#Passing the testing sample codes through TfidVectorizer

y_pred = rm.predict(x_test_tf)  #Getting prediction of RandomForestClassifier
y_pred_cnn = mlp.predict(x_test_tf) #Getting prediction of Multilayer Perceptron
y_pred_nb = best_mnb_model.predict(x_test_tf)  #Getting prediction of Multinomial Naive Bayes

#Function for reading a file [Upload file function] 
def read_file(open_file):   #Takes a file as input
    with open(open_file, 'r') as file:  #Open the passed file in read mode
        read_content = file.read()  #Reads the content of the file
        Testing(read_content)   #Passes the read content into the Testing function to get the prediction



#Function to get the predicition 
def Testing(test_code): #Takes code as input
  avg = []  #Creates list to store predictions and get the best prediction
  also_detected = []
  test_code = vectorizer.transform([test_code]) #Passes the given code through vectorizer
  pred_lang1 = (rm.predict(test_code)[0])  #Gets prediction of RandomForestClassifier
  avg.append(pred_lang1) #Appends prediction to list
  pred_lang2 = (best_mnb_model.predict(test_code)[0]) #Gets prediction of Multinomial Naive Bayes
  avg.append(pred_lang2) #Appends prediction to list
  pred_lang3 = (mlp.predict(test_code)[0]) #Gets prediction of Multilayer Perceptron [Convolutional Neural Network]
  avg.append(pred_lang3) #Appends prediction to list
  answer = max(Counter(avg), key=Counter(avg).get)  #Gets best prediction out of the three predictions
  for x in avg:
      if x != answer:
          also_detected.append(x)
          
  print(f"""
        \n\n
        Detected Language : {answer}  
        \n        
        """)
  
  if len(also_detected) > 0:
    print("Other potential programming languages detected are : ")
    for y in also_detected:   
        print("-",y)
    print("\n\n")



def FilePathReading(Code_file_path):
    with open(Code_file_path , "r") as file:
        Code_file_read = file.read()
        Testing(Code_file_read)
        
running = True

while running:
    file_path_code = input("Please enter the file path // Enter 'exit' to exit the terminal : ")
    if file_path_code != 'exit'.lower():
        FilePathReading(file_path_code)
    else:
        running = False 