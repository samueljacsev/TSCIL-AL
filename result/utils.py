import csv
import os

def save_acc_to_csv(accs_data, run, task, cycle, filename="output.csv"):
    """
    Saves a list of values to a CSV file with additional columns for run, task, and cycle.

    Parameters:
        data (list): A list of numerical values to be saved as task columns.
        run (int): The run identifier.
        task (int): The task identifier.
        cycle (int): The cycle identifier.
        filename (str): The name of the CSV file to save the data. Default is 'output.csv'.
    """
    # Generate column names for the tasks
    task_columns = [f"task_{i+1}" for i in range(len(accs_data))]

    # Check if the file exists
    file_exists = os.path.isfile(filename)

    # Open the file in append mode
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write the header if the file does not exist
        if not file_exists:
            print('task_columns:', task_columns)
            header = ["run", "task", "cycle"] + task_columns
            writer.writerow(header)

        # Write the data row
        print('accs_data:', accs_data)
        row = [run, task, cycle] + accs_data.tolist()
        writer.writerow(row)