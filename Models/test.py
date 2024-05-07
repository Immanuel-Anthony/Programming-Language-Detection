import json

def combine_json_files(file1, file2, output_file):
    # Read the content of the first JSON file
    with open(file1, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)

    # Read the content of the second JSON file
    with open(file2, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)

    # Check if both data are dictionaries
    if isinstance(data1, dict) and isinstance(data2, dict):
        # Combine the data from both dictionaries
        combined_data = {**data1, **data2}
    elif isinstance(data1, list) and isinstance(data2, list):
        # Combine the data from both lists
        combined_data = data1 + data2
    else:
        # Handle other cases or raise an error based on your specific needs
        raise ValueError("Unsupported data types. Both files should contain either dictionaries or lists.")

    # Write the combined data to the output file
    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(combined_data, output, indent=2)

# Example usage:
combine_json_files('final_ds.json', 'D.json', 'final_ds1.json')
