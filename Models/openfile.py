import tkinter as tk
from tkinter import filedialog
import os

def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        print(f"Content of {file_path}:\n")
        print(content)

def main():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Ask the user to select a file using a file dialog
    file_path = filedialog.askopenfilename(title="Select a File", filetypes=[("Text Files", "*.txt")])

    if file_path:
        # Read and print the content of the selected file
        read_file(file_path)
    else:
        print("No file selected.")

if __name__ == "__main__":
    main()
