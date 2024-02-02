import os
from xml.etree.ElementTree import XMLParser 
import pandas as pd
from tqdm import tqdm

combined_data_df = pd.DataFrame()

# Define the path to the dataset directory
dataset_path = "./dataset/smaller_file/"

# Get the list of XML files in the dataset directory
xml_files = [filename for filename in os.listdir(dataset_path) if filename.endswith(".xml")]
i = 0
# Initialize a tqdm progress bar
for filename in tqdm(xml_files, desc="Processing XML files", unit="file"):
    XMLP = XMLParser(os.path.join(dataset_path, filename))
    XMLP.parse_xml()
    XMLP.find_all_records()
    XMLP.extract_specific_datafields()
    XMLP.dropnolcc()
    specific_datafields_df = XMLP.remove_na()
    i += specific_datafields_df.shape[0]
    combined_data_df = pd.concat([combined_data_df, specific_datafields_df], ignore_index=True)

# Reset index to avoid potential issues
combined_data_df = combined_data_df.reset_index(drop=True)

# Save the combined DataFrame to XML
combined_data_df.to_xml("./dataset/lcc_big_dataset.xml", index=False, encoding="utf-8")

if __name__ == '__main__':
    pass
