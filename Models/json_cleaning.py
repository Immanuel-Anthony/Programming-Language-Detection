import ijson
import json

def clean_noisy_data(data, outlier_threshold=3, replace_value=None):
    """
    Recursively clean noisy data in a JSON-like data structure by handling outliers
    in numeric data.

    Parameters:
    - data: The JSON-like data structure.
    - outlier_threshold: The threshold to identify outliers.
    - replace_value: The value to replace outliers with.

    Returns:
    - The cleaned data structure.
    """
    if isinstance(data, dict):
        cleaned_data = {}
        for key, value in data.items():
            cleaned_value = clean_noisy_data(value, outlier_threshold, replace_value)
            cleaned_data[key] = cleaned_value
        return cleaned_data
    elif isinstance(data, list):
        return [clean_noisy_data(item, outlier_threshold, replace_value) for item in data]
    elif isinstance(data, (int, float)):
        # Handle numeric data (replace outliers)
        if abs(data) > outlier_threshold:
            return replace_value
        else:
            return data
    else:
        return data


def preprocess_large_json(input_file, output_file, chunk_size=1000, outlier_threshold=3, replace_value=None):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w') as outfile:
            # Create an ijson parser for the input file
            parser = ijson.items(infile, 'item')

            # Iterate through chunks of the JSON data
            for chunk_index, chunk in enumerate(parser):
                try:
                    cleaned_chunk = clean_noisy_data(chunk, outlier_threshold, replace_value)
                except Exception as e:
                    print(f"Error cleaning chunk {chunk_index + 1}: {e}")
                    continue

                # Write the cleaned chunk to the output file
                json.dump(cleaned_chunk, outfile, indent=2)
                outfile.write('\n')  # Add a newline between chunks

                if (chunk_index + 1) % chunk_size == 0:
                    print(f"Processed {chunk_index + 1} chunks")

            print(f"Preprocessing successful. Cleaned data saved to {output_file}")

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        print("Attempting to handle trailing garbage...")

        # Try to remove any trailing non-JSON content by reading the remaining lines
        for remaining_line in infile:
            try:
                cleaned_chunk = json.loads(remaining_line)
                json.dump(cleaned_chunk, outfile, indent=2)
                outfile.write('\n')
            except json.JSONDecodeError:
                print(f"Ignoring non-JSON line: {remaining_line.strip()}")

        print(f"Preprocessing successful. Cleaned data saved to {output_file}")

    except Exception as e:
        print(f"Error: {e}")

# Example usage
input_file_path = 'Go_output.json'
output_file_path = 'Go_output_final.json'
preprocess_large_json(input_file_path, output_file_path, outlier_threshold=30, replace_value=None)

