import json
import re

def clean_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        # Read the content of the JSON file
        json_content = f.read()

        # Remove single-line comments
        json_content = re.sub(r'//.*', '', json_content)

        # Remove multi-line comments
        json_content = re.sub(r'/\*.*?\*/', '', json_content, flags=re.DOTALL)

        # Remove leading and trailing whitespaces from each line
        json_content = '\n'.join(line.strip() for line in json_content.splitlines())

        # Remove extra whitespace between JSON elements
        json_content = re.sub(r'\s*([:,{}])\s*', r'\1', json_content)

        # Parse the cleaned JSON
        cleaned_json = json.loads(json_content)

        print(cleaned_json)
        
    with open(output_file, 'w') as f:
        # Write the cleaned JSON to the output file
        json.dump(cleaned_json, f, indent=2)

# Example usage:
clean_json('Go_output.json', 'Go_output_final.json')
