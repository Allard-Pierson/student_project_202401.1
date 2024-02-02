import pandas as pd

class DataFinder:
    def __init__(self, data):
        self.data = data

    def find_matching_items(self, value):
        # Remove None values
        value = [x for x in value if x is not None]
        # Initialize an empty DataFrame to store results
        data = pd.DataFrame()
        for i in value:
            # Append the rows that contain the matching value
            matching_rows = self.data[self.data['LCC_classificatie'].str.contains(i, na=False)]
            data = data.append(matching_rows, ignore_index=True)
        return data
