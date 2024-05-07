import os
import json

def write_to_json(data, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=2)

def read_code_files(folder_path, language, max_files=200):
    code_data = []

    file_count = 0  # Keep track of the number of files read
    for filename in os.listdir(folder_path):
        if file_count >= max_files:
            break

        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                code_snippet = file.read()
                code_data.append({
                    'language': language,
                    'code': code_snippet
                })
                file_count += 1

    return code_data

if __name__ == "__main__":
    main_folder_path = r"C:\Users\immanuel\Desktop\projekt\Programming Language Detection\DS_35"
    output_file = "200_output.json"
    max_files_per_folder = 500  # Set the maximum number of files to read from each folder

    code_data = []
    for language_folder in os.listdir(main_folder_path):
        language_folder_path = os.path.join(main_folder_path, language_folder)
        if os.path.isdir(language_folder_path):
            code_data.extend(read_code_files(language_folder_path, language_folder, max_files_per_folder))

    write_to_json(code_data, output_file)

    print(f"Data from code files in '{main_folder_path}' has been written to '{output_file}'.")
