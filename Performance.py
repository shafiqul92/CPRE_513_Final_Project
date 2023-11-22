import pandas as pd


def find_match(rounded_value, actual_value, output_paths, actual_df):
    """Check for a match in actual_df for the rounded value with exactly matching paths in output_paths."""
    if abs(rounded_value - actual_value) > 0.5:  # Adjust this threshold as needed
        return False
    actual_rows = actual_df[actual_df['Input'] == rounded_value]
    for _, actual_row in actual_rows.iterrows():
        actual_paths = parse_paths(actual_row['Path'], as_list=True)
        if actual_paths == output_paths:
            return True
    return False



def parse_paths(path, as_list=False):
    """Convert paths to lists (or sets) of integers, handling various formats."""
    if isinstance(path, int):
        return [path] if as_list else {path}
    elif isinstance(path, (str, list, set)):
        if isinstance(path, str) and path.startswith('{') and path.endswith('}'):
            path = path[1:-1]  # Strip the curly braces
        elements = map(int, path.split(';')) if isinstance(path, str) else path
        return list(elements) if as_list else set(elements)
    else:
        raise ValueError(f"Unexpected path format: {path}")


def calculate_correct_path_percentage(actual_path, output_path):
    """Calculate the percentage of correct path elements."""
    actual_elements = actual_path.split(';')
    output_elements = output_path.split(';')

    shortest_length = min(len(actual_elements), len(output_elements))
    correct_matches = sum(a == o for a, o in zip(actual_elements[:shortest_length], output_elements[:shortest_length]))

    longest_length = max(len(actual_elements), len(output_elements))
    if longest_length == 0:
        return 100

    return (correct_matches / longest_length) * 100


# Read the CSV files for output and actual test cases
# Replace the paths below with the actual file paths
output_df = pd.read_csv('/path/to/output.csv')  # Replace with the actual path to the output CSV
actual_df = pd.read_csv('/path/to/test_case0.csv')  # Replace with the actual path to the test case CSV

# Convert the 'Input' column in actual_df to integers for comparison
actual_df['Input'] = actual_df['Input'].astype(int)
output_df['Path'] = output_df['Path'].apply(lambda p: parse_paths(p, as_list=True))
actual_df['Path'] = actual_df['Path'].apply(lambda p: parse_paths(p, as_list=True))

correct_count = 0
incorrect_count = 0
total_correct_path_percentage = 0
total_count = len(output_df)

# Iterate over each row in output_df for comparison
for _, row in output_df.iterrows():
    actual_value = row['Input']
    rounded_value = round(actual_value)
    output_paths = row['Path']

    if find_match(rounded_value, actual_value, output_paths, actual_df):
        correct_count += 1
    else:
        incorrect_count += 1

    # Calculate the correct path percentage for each row
    actual_path = ';'.join(map(str, actual_df.loc[actual_df['Input'] == rounded_value, 'Path'].iloc[0]))
    output_path = ';'.join(map(str, output_paths))
    correct_path_percentage = calculate_correct_path_percentage(actual_path, output_path)
    total_correct_path_percentage += correct_path_percentage

# Calculate the percentages
percentage_correctness = (correct_count / total_count) * 100
percentage_incorrectness = (incorrect_count / total_count) * 100
average_correct_path_percentage = total_correct_path_percentage / total_count

# Print the results
print(f"Total Entries: {total_count}")
print(f"Correct Matches: {correct_count}")
print(f"Incorrect Matches: {incorrect_count}")
print(f"Percentage Correctness: {percentage_correctness:.2f}%")
print(f"Percentage Incorrectness: {percentage_incorrectness:.2f}%")
print(f"Average Correct Path Percentage: {average_correct_path_percentage:.2f}%")
print(f"Total Correct Path Percentage: {total_correct_path_percentage:.2f}%")
