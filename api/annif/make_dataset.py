"""
This file contains functions to parse different MARC XML files to make datasets suitable for Annif.
Addtionally It contains a function for cleaning up the data.

"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from tqdm import tqdm
import langid

def ubxml_to_titles(xml_files, output_csv):
    ''' 
    This function takes a list of book MARC XML files (from UB) and an output_csv target file
    and returns the titles & subjects into a single CSV.
    '''

    # Makes dictionary for the text and a separate column for each subject
    data = {'Content': [], 'Subject1': [], 'Subject2': [], 'Subject3': [], 'Subject4': [], 'Subject5': [], 
            'Subject6': [], 'Subject7': [], 'Subject8': [], 'Subject9': [], 'Subject10': [], 'Subject11': [],
            'Subject12': [], 'Subject13': []}

    # Use tqdm to create a progress bar
    for xml_file in tqdm(xml_files, desc="Processing XML files", unit="file"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Iterate through each record in the XML
        for record in root.findall(".//record"):

            # The summaries are found in tag 520
            title = record.find(".//datafield[@tag='245']/subfield[@code='a']")

            # The subjects are found in tag 650. They should be LCSH subjects, so ind2='0'.
            subject = record.find(".//datafield[@tag='650'][@ind2='0']/subfield[@code='a']") # Checks if atleast one can be found
            
            # If this record contains both a title and subjects, append to the data dictionary
            if title is not None and subject is not None:

                # Finds all subjects
                subjects_list = record.findall(".//datafield[@tag='650'][@ind2='0']/subfield[@code='a']")    

                data['Content'].append(title.text)

                # Update subject columns
                for i in range(len(data) - 1):
                    if i < len(subjects_list):
                        data[f'Subject{i + 1}'].append(subjects_list[i].text)
                    else:
                        data[f'Subject{i + 1}'].append(None)  # or use np.nan if using NumPy

    # Create a DataFrame from the data dictionary
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)


def snxml_to_titles(xml_file, output_csv):
    ''' 
    This function takes a MARC XML file (from Springer Nature) and an output_csv target file
    and returns the titles & subjects into a single CSV.
    '''
    prefix = {'marc': 'http://www.loc.gov/MARC21/slim'}
    
    # Makes dictionary for the text and a separate column for each subject
    data = {'Content': [], 'Subject1': [], 'Subject2': [], 'Subject3': [], 'Subject4': [], 'Subject5': [], 
            'Subject6': [], 'Subject7': [], 'Subject8': [], 'Subject9': [], 'Subject10': [], 'Subject11': [],
            'Subject12': [], 'Subject13': []}

    # Use tqdm to create a progress bar
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Iterate through each record in the XML
    for record in root.findall(".//marc:record", namespaces=prefix):

        # The summaries are found in tag 520
        title = record.find(".//marc:datafield[@tag='245']/marc:subfield[@code='a']", namespaces=prefix)

        # The subjects are found in tag 650. They should be LCSH subjects, so ind2='0'.
        subject = record.find(".//marc:datafield[@tag='650'][@ind2='0']/marc:subfield[@code='a']", namespaces=prefix) # Checks if atleast one can be found # Checks if atleast one can be found
        
        # If this record contains both a title and subjects, append to the data dictionary
        if title is not None and subject is not None:

            # Finds all subjects
            subjects_list = record.findall(".//marc:datafield[@tag='650'][@ind2='0']/marc:subfield[@code='a']", namespaces=prefix)    

            data['Content'].append(title.text)

            # Update subject columns
            for i in range(len(data) - 1):
                if i < len(subjects_list):
                    data[f'Subject{i + 1}'].append(subjects_list[i].text)
                else:
                    data[f'Subject{i + 1}'].append(None)  # or use np.nan if using NumPy

    # Create a DataFrame from the data dictionary
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)


def ubxml_to_summaries(xml_files, output_csv):
    ''' 
    This function takes a list of book MARC XML files and an output_csv target file
    and returns the summaries & subjects into a single CSV.
    '''
    # Makes dictionary for the text and a separate column for each subject
    data = {'Content': [], 'Subject1': [], 'Subject2': [], 'Subject3': [], 'Subject4': [], 'Subject5': [], 
            'Subject6': [], 'Subject7': [], 'Subject8': [], 'Subject9': [], 'Subject10': [], 'Subject11': [],
            'Subject12': [], 'Subject13': []}

    # Use tqdm to create a progress bar
    for xml_file in tqdm(xml_files, desc="Processing XML files", unit="file"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Iterate through each record in the XML
        for record in root.findall(".//record"):

            # The summaries are found in tag 520
            title = record.find(".//datafield[@tag='245']/subfield[@code='a']")
            summary = record.find(".//datafield[@tag='520']/subfield[@code='a']")

            # The subjects are found in tag 650. They should be LCSH subjects, so ind2='0'.
            subject = record.find(".//datafield[@tag='650'][@ind2='0']/subfield[@code='a']") # Checks if atleast one can be found
            
            # If this record contains both a title and subjects, append to the data dictionary
            if title is not None and summary is not None and subject is not None:

                # Finds all subjects
                subjects_list = record.findall(".//datafield[@tag='650'][@ind2='0']/subfield[@code='a']")

                content = title.text + ' | ' + summary.text
                data['Content'].append(content)

                # Update subject columns
                for i in range(len(data) - 1):
                    if i < len(subjects_list):
                        data[f'Subject{i + 1}'].append(subjects_list[i].text)
                    else:
                        data[f'Subject{i + 1}'].append(None)  # or use np.nan if using NumPy

    # Create a DataFrame from the data dictionary
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)


def snxml_to_summaries(xml_file, output_csv):
    ''' 
    This function takes a MARC XML file (from Springer Nature) and an output_csv target file
    and returns the summaries & subjects into a single CSV.
    '''

    prefix = {'marc': 'http://www.loc.gov/MARC21/slim'}

    # Makes dictionary for the text and a separate column for each subject
    data = {'Content': [], 'Subject1': [], 'Subject2': [], 'Subject3': [], 'Subject4': [], 'Subject5': [], 
            'Subject6': [], 'Subject7': [], 'Subject8': [], 'Subject9': [], 'Subject10': [], 'Subject11': [],
            'Subject12': [], 'Subject13': []}

    # Use tqdm to create a progress bar
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Iterate through each record in the XML
    for record in root.findall(".//marc:record", namespaces=prefix):

        # The summaries are found in tag 520 and titles in 245
        title = record.find(".//marc:datafield[@tag='245']/marc:subfield[@code='a']", namespaces=prefix)
        summary = record.find(".//marc:datafield[@tag='520']/marc:subfield[@code='a']", namespaces=prefix)

        # The subjects are found in tag 650. They should be LCSH subjects, so ind2='0'.
        subject = record.find(".//marc:datafield[@tag='650'][@ind2='0']/marc:subfield[@code='a']", namespaces=prefix) # Checks if atleast one can be found
        
        # If this record contains both a title and subjects, append to the data dictionary
        if title is not None and summary is not None and subject is not None:

            # Finds all subjects
            subjects_list = record.findall(".//marc:datafield[@tag='650'][@ind2='0']/marc:subfield[@code='a']", namespaces=prefix)

            content = title.text + ' | ' + summary.text
            data['Content'].append(content)

            # Update subject columns
            for i in range(len(data) - 1):
                if i < len(subjects_list):
                    data[f'Subject{i + 1}'].append(subjects_list[i].text)
                else:
                    data[f'Subject{i + 1}'].append(None)  # or use np.nan if using NumPy

    # Create a DataFrame from the data dictionary
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)


def clean_dataset(dataset, clean_dataset):
    '''
    Takes a raw dataset, removes duplicates and unnecessary symbols, removes non-english text, 
    returns new dataset and examples during cleaning.

    '''
    df = pd.read_csv(dataset)
    print("Original size: ", df.shape)

    orginal_data = df.head() # Saves data preview of original data
    
    # Removes rows where there is a duplicate in the content (title or summary)
    print("Removing duplicates...")
    df = df.drop_duplicates(subset='Content')

    print("New size: ", df.shape)

    # Removes unnecessary '/', '"' & ':' symbols
    print("Removing unnecessary symbols...")
    df['Content'] = df['Content'].str.replace('"', '')
    df['Content'] = df['Content'].str.replace('/', '').str.replace(':', '')

    clean_data_step1 = df.head() # Saves data preview of first cleaning step

    # Detects what language each book is in
    print("Detecting English language... (may take a while)")
    df['Language'], df['Confidence'] = zip(*df['Content'].apply(detect_language).tolist())

    # Remove all columns which are not in english.
    df = df[df['Language'] == 'en'] 
    print("Removing non-English text...")
    # Remove the columns Language and Confidence
    df = df.drop(['Language','Confidence'],axis=1) 

    clean_data_step2 = df.head() # Saves data preview of second cleaning step
    print("Final size: ", df.shape)

    # Converts new data to .csv file
    print("Converting to csv...")
    df.to_csv(clean_dataset, index=False)

    return orginal_data, clean_data_step1, clean_data_step2


def detect_language(text):
    ''' Takes a text and returns the classified language and the confidence score'''
    lang, confidence = langid.classify(text)
    return lang, confidence


def shift_subjects(input_tsv, output_tsv):
    ''' 
    This function takes an input- and output .tsv filepath.
    It shifts all the subjects to the left
    Returns the altered dataset and before and after of the first few entries.
    
    '''

    df = pd.read_csv(input_tsv, sep='\t') # Reads .tsv file
    original_data = df.head() # Saves data preview of orignal data
    df = df.fillna('') # Fills the NaN values with an empty string

    num_subject_columns = len(df.columns) - 1  

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Iterate over each subject column
        for i in range(2, num_subject_columns):
            current_subject = f'Subject{i}_URI'

            # Find the leftmost empty column
            empty_columns = [f'Subject{j}_URI' for j in range(1, i) if row[f'Subject{j}_URI'] == '']

            # If the current subject is non-empty and there is an empty column, move the subject
            if row[current_subject] != '' and empty_columns:
                # Move the non-empty subject to the leftmost empty column
                row[empty_columns[0]], row[current_subject] = row[current_subject], ''

    new_data = df.head() # Saves data preview of new data

    # Exports to .tsv without header and index.     
    df.to_csv(output_tsv, sep='\t', header=False, index=False)

    return original_data, new_data
