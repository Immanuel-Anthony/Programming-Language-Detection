import os
import json

def read_code_files(folder_path, language):
    code_data = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                code_snippet = file.read()
                code_data.append({
                    'language': language,
                    'code': code_snippet
                })

    return code_data

def write_to_json(data, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=2)

if __name__ == "__main__":
    main_folder_path = r"C:\Users\immanuel\Desktop\projekt\Programming Language Detection\Initial - DS - 35 - Copy"  # Replace with the actual path to your "Train" folder
    output_file = "huge_output.json"  # You can change the output file name if needed

    # Iterate through subfolders
    code_data = []
    for language_folder in os.listdir(main_folder_path):
        language_folder_path = os.path.join(main_folder_path, language_folder)
        if os.path.isdir(language_folder_path):
            code_data.extend(read_code_files(language_folder_path, language_folder))

    # Write data to JSON file
    write_to_json(code_data, output_file)

    print(f"Data from code files in '{main_folder_path}' has been written to '{output_file}'.")
