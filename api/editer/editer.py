import pandas as pd
import os

class XMLEDITER:
    def __init__(self, xml_file):
        if not os.path.exists(xml_file):
            raise FileNotFoundError(f"File not found: {xml_file} you are at {os.getcwd()}")
        self.xml_file = xml_file

    def open_xml(self):
        self.df = pd.read_xml(self.xml_file)

    def remove_na(self):
        # Fill NaN values in the DataFrame with 0 (optional)
        self.df.fillna("")
        return self.df
    
    def dropnolcc(self):
        # Remove all rows that do not have a LCC classification
        self.df = self.df[self.df['LCC_classificatie'].notna()]
        return self.df