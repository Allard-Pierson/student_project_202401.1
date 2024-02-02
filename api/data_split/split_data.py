import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import threading
import pandas as pd

class XMLSplitter:
    def __init__(self, large_xml_file, output_folder):
        self.large_xml_file = large_xml_file
        self.output_folder = output_folder
        self.tree = ET.parse(large_xml_file)
        self.root = self.tree.getroot()
        self.total_elements = len(self.root.findall('.//record'))
        self.split_counter = 1

    def open_xml(self):
        self.df = pd.read_xml(self.xml_file)

    def write_element_to_file(self, element, file_path):
        with open(file_path, 'wb') as file:
            file.write(ET.tostring(element))

    def process_batch(self, batch, batch_number, semaphore):
        try:
            semaphore.acquire()
            smaller_file_path = os.path.join(self.output_folder, f'smaller_file_{batch_number}.xml')

            # Create a new root element for the smaller XML file
            smaller_root = ET.Element('root')

            # Add the elements to the smaller root
            for elem in batch:
                smaller_root.append(elem)

            # Create a new tree and write it to the smaller XML file
            smaller_tree = ET.ElementTree(smaller_root)
            smaller_tree.write(smaller_file_path)
        finally:
            semaphore.release()

    def create_files(self, max_threads: int, batch_size: int):
        # Create the output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

        # Create a semaphore to control the number of active threads
        semaphore = threading.Semaphore(max_threads)

        # Iterate through the XML elements with a progress bar
        for i in tqdm(range(0, self.total_elements, batch_size), desc="Splitting XML"):
            # Add the elements to the list for this batch
            batch = self.root[i:i + batch_size]

            # Create a new thread to process this batch
            thread = threading.Thread(target=self.process_batch, args=(batch, self.split_counter, semaphore))
            thread.start()
            thread.join()

            # Increment the counter for the next batch
            self.split_counter += 1
    
    def parse_xml(self):
        with open(self.xml_file, 'r', encoding='utf-8') as file:
            xml_content = file.read()
        parser = ET.XMLParser(encoding="utf-8")
        # Parse the XML content and store it in the 'data' attribute (repeated method, consider removing one)
        self.data = ET.fromstring(xml_content, parser=parser)

    def extract_record_data(self, record_element):
        record_data = {}
        for datafield in record_element.findall(".//datafield"):
            tag = datafield.get("tag")
            subfields = self.extract_subfields(datafield)
            for subfield_tag, subfield_text in subfields.items():
                col_name = f"datafield_{tag}_{subfield_tag}"
                record_data[col_name] = subfield_text
        return record_data

    def find_all_records(self):
        if self.data is not None:
            records = self.data.findall('.//record')
            record_data_list = []
            for record in records:
                record_data_list.append(self.extract_record_data(record))
            # Create a DataFrame from the extracted record data
            self.df = pd.DataFrame(record_data_list)

            return self.df
        return []

    def extract_subfields(self, datafield):
        subfields = {}
        for subfield in datafield:
            code = subfield.get("code")
            text = subfield.text
            subfields[code] = text
        return subfields

    def extract_specific_datafields(self):
        specific_datafields = ['datafield_041_a', 'datafield_044_a',
                                'datafield_090_a', 'datafield_100_a', 'datafield_245_a',
                                'datafield_260_a', 'datafield_300_a', 'datafield_490_a',
                                'datafield_651_a', 'datafield_655_a']

        if hasattr(self, 'df'):
            # Check if columns in specific_datafields exist in self.df

            valid_columns = [col for col in specific_datafields if col in self.df.columns]
            specific_datafields_df = self.df[valid_columns].copy()

            # Dictionary met de oude namen als sleutels en nieuwe namen als waarden
            rename_dict = {
                'datafield_041_a': 'Taal_van_het_boek',
                'datafield_044_a': 'Land_van_publicatie',
                'datafield_090_a': 'LCC_classificatie',
                'datafield_100_a': 'Auteur_UID',
                'datafield_245_a': 'Titel_en_ondertitel',
                'datafield_260_a': 'Uitgever_plaats_publicatiejaar',
                'datafield_300_a': 'Aantal_paginas',
                'datafield_490_a': 'Boekenreeks',
                'datafield_651_a': 'geografisch_onderwerp',
                'datafield_655_a': 'genre'
            }
            # Hernoem de kolommen
            specific_datafields_df.rename(columns=rename_dict, inplace=True)
            self.df = specific_datafields_df
        else:
            print("Please call find_all_records() before extracting specific datafields.")
            return pd.DataFrame()