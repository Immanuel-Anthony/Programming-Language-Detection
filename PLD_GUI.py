import tkinter as tk
from tkinter import filedialog
from threading import Thread
import requests
import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from collections import Counter

# Define the raw URL of the JSON file on your public repository
JSON_URL = 'https://gist.githubusercontent.com/Arshad-ashuu/ec5672a56f13c26216bcf3426fbf0a8b/raw/5339643ed5527678f8a87ee8cc22ddb4b95d8431/gistfile1.txt'

def load_data():
    global dt, X, Y, x_train, x_test, y_train, y_test, vectorizer, rm, mnb, mlp, x_test_tf, acc, acc1, acc2, best_params_mnb, best_mnb_model
    try:
        # Download the JSON content from the gist repository URL
        response = requests.get(JSON_URL)
        response.raise_for_status()  # Raise an exception for bad response status
        dt = pd.read_json(io.StringIO(response.text))
    except requests.exceptions.RequestException as e:
        # Handle any exceptions that may occur during the download
        print(f"Error downloading JSON file: {e}")
        return

    X, Y = dt.code, dt.language
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    pattern = r"""\b[A-Za-z_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]`"']"""
    vectorizer = TfidfVectorizer(token_pattern=pattern)

    # Fit the vectorizer on the training data
    x_train_tf = vectorizer.fit_transform(x_train)

    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1500,batch_size=64)  
    mnb = MultinomialNB(alpha=0.1)  
    rm = RandomForestClassifier(max_depth = None, random_state = 42, n_estimators=100) 

    param_grid_mnb = {'alpha': [0.1, 0.5, 1.0], 'fit_prior': [True, False]}
    grid_search_mnb = GridSearchCV(mnb, param_grid_mnb, cv=5, scoring='accuracy')

    grid_search_mnb.fit(x_train_tf, y_train)
    best_params_mnb = grid_search_mnb.best_params_
    best_mnb_model = grid_search_mnb.best_estimator_

    rm.fit(x_train_tf, y_train)
    mlp.fit(x_train_tf, y_train)
    best_mnb_model.fit(x_train_tf, y_train)

    x_test_tf = vectorizer.transform(x_test)

    y_pred = rm.predict(x_test_tf)
    y_pred_cnn = mlp.predict(x_test_tf)
    y_pred_nb = best_mnb_model.predict(x_test_tf)

    global acc, acc1, acc2

    acc = accuracy_score(y_test, y_pred)
    acc1 = accuracy_score(y_test, y_pred_cnn)
    acc2 = accuracy_score(y_test, y_pred_nb)

def read_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        with open(file_path, 'r') as file:
            read_content = file.read()
            Testing(read_content)

def Testing(test_code):
    global best_mnb_model, vectorizer
    avg = []  #Creates list to store predictions and get the best prediction
    
    loading_thread.join()
    
    test_code = vectorizer.transform([test_code]) #Passes the given code through vectorizer
    pred_lang1 = (rm.predict(test_code)[0])  #Gets prediction of RandomForestClassifier
    avg.append(pred_lang1) #Appends prediction to list
    pred_lang2 = (best_mnb_model.predict(test_code)[0]) #Gets prediction of Multinomial Naive Bayes
    avg.append(pred_lang2) #Appends prediction to list
    pred_lang3 = (mlp.predict(test_code)[0]) #Gets prediction of Multilayer Perceptron [Convolutional Neural Network]
    avg.append(pred_lang3) #Appends prediction to list
    answer = max(Counter(avg), key=Counter(avg).get)
    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"The detected Code is: {answer}\n")

    result_text.config(state=tk.DISABLED)

def on_load_button_click():
    load_data()
    load_button.config(state=tk.NORMAL)

root = tk.Tk()
root.title("Poland")

# Configure window attributes
s_w = root.winfo_screenwidth()
s_h = root.winfo_screenheight()
width = 700
height = 800
x_c = (s_w - width) // 2
x_h = (s_h - 100 - height) // 2
root.geometry(f"{width}x{height}+{x_c}+{x_h}")
root.resizable(False, False)

# Background
canvas = tk.Canvas(root, width=700, height=800)
canvas.pack()

color1 = (165, 82, 227)
color2 = (203, 170, 227)

window_height = 800
for i in range(window_height):
    r = int(color1[0] + (color2[0] - color1[0]) * i / window_height)
    g = int(color1[1] + (color2[1] - color1[1]) * i / window_height)
    b = int(color1[2] + (color2[2] - color1[2]) * i / window_height)

    hex_color = f'#{r:02x}{g:02x}{b:02x}'
    canvas.create_rectangle(0, i, 700, i + 1, fill=hex_color, outline="")

center_x = 700 // 2

code_label = tk.Label(root, text="Poland", bg='#a656e3', fg="black", font=("Helvetica", 20, "bold"))
code_label.place(x=center_x, y=30, anchor='center')

code_label1 = tk.Label(root, text="An ultimate tool for PrOgram LANguage Detection", bg='#a85ae3', fg="black",
                       font=("Helvetica", 17))
code_label1.place(x=center_x, y=68, anchor='center')

code_label = tk.Label(root, text="Enter Code:", bg='#a95de3', fg="black", font=("Helvetica", 12, "bold"))
code_label.place(x=20, y=83)

code_entry = tk.Text(root, height=20, width=92, bg="#ecf0f1", font=("Helvetica", 10), borderwidth=2, relief="groove",
                     padx=5, pady=5)
code_entry.place(x=20, y=111)

test_button = tk.Button(root, text="Test Code", command=lambda: Testing(code_entry.get("1.0", tk.END)), bg="#2ecc71",
                        fg="black", font=("Helvetica", 15), borderwidth=2, relief="raised", padx=10, pady=5, bd=3,
                        cursor="hand2")

test_button.place(x=20, y=522)

load_button = tk.Button(root, text="Load File", command=read_file, state=tk.DISABLED,
                        bg="#e74c3c", fg="black", font=("Helvetica", 15), borderwidth=2, relief="raised", padx=10,
                        pady=5, bd=3, cursor="hand2")
load_button.place(x=180, y=522)

result_text = tk.Text(root, wrap="word", height=1, width=41, state=tk.DISABLED, bg="#ecf0f1", font=("Helvetica", 20),
                      borderwidth=2, relief="groove", padx=20, pady=5)
result_text.place(x=20, y=462)

code_label2 = tk.Label(root, text="Languages Supported", bg='#c295e3', fg="black", font=("Helvetica", 20, "bold"))
code_label2.place(x=center_x, y=602, anchor='center')

langs1 = ["Ada", "C", "C#", "C++", "Java", "JavaScript", "Julia", "Kotlin", "TCL"]
langs2 = ["Mathematica", "MATLAB", "Perl", "PHP", "Powershell", "Python", "Ruby", "Smalltalk", ]
langs3 = ["Rust", "Swift", "UNIX SHELL", "COBOL", "dart", "Fortran", "Groovy", "R"]
langs4 = ["Haskell", "Lisp", "HTML", "Go", "Erlang", "Lua", "Prolog", "Scheme", "D"]
langs5 = ["Coming Soon"]

formatted_text = "\n".join([f"• {name:<9}" for name in langs1])
lang1_label = tk.Label(root, text=formatted_text, justify='left', bg='#c99dfa', font=("Helvetica", 10), borderwidth=2,
                       relief="groove", padx=20, pady=5)
lang1_label.place(x=20, y=630)
formatted_text2 = "\n".join([f"• {name:<9}" for name in langs2])
lang2_label = tk.Label(root, text=formatted_text2, justify='left', bg='#c99dfa', font=("Helvetica", 10), borderwidth=2,
                       relief="groove", padx=20, pady=5)
lang2_label.place(x=140, y=630)
formatted_text3 = "\n".join([f"• {name:<9}" for name in langs3])
lang3_label = tk.Label(root, text=formatted_text3, justify='left', bg='#c99dfa', font=("Helvetica", 10), borderwidth=2,
                       relief="groove", padx=20, pady=5)
lang3_label.place(x=280, y=630)
formatted_text4 = "\n".join([f"• {name:<9}" for name in langs4])
lang4_label = tk.Label(root, text=formatted_text4, justify='left', bg='#c99dfa', font=("Helvetica", 10), borderwidth=2,
                       relief="groove", padx=20, pady=5)
lang4_label.place(x=410, y=630)
formatted_text5 = "\n".join([f"• {name:<9}" for name in langs5])
lang5_label = tk.Label(root, text=formatted_text5, justify='left', bg='#c99dfa', font=("Helvetica", 10), borderwidth=2,
                       relief="groove", padx=20, pady=10)
lang5_label.place(x=530, y=630)

# Load data asynchronously
loading_thread = Thread(target=load_data)
loading_thread.start()

# Enable load button when loading is complete
loading_thread.join()
load_button.config(state=tk.NORMAL)

root.mainloop()
