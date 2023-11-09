import csv
from utils import convert_to_numbers

class my_data_frame:

    def __init__(self, file_name) -> None:
        self.file_name = file_name
        result = self.csv_read(file_name)
        if 'Error' in result:
           raise Exception(result['Error'])
        else:
            self.columns = result['Columns']
            self.data = result['Data']
  
    def csv_read(self, file_name):
        result = {}

        try:
            with open(file_name, 'r', newline='') as csv_file:
                csv_reader = csv.reader(csv_file)
                columns = next(csv_reader)
                result['Columns'] = columns
                data = {}
                for column in columns:
                    data[column] = []
                for row in csv_reader:
                    for idx, column in enumerate(columns):
                        if row[idx] != "":
                            data[column].append(row[idx])
                        else:
                            data[column].append(None)
                for column in columns:
                    data[column] = convert_to_numbers(data[column])
                result['Data'] = data
        
        except FileNotFoundError:
            result['Error'] = f"File '{file_name}' not found."
        except Exception as e:
            result['Error'] = f"An error occurred: {e}"
        return result
