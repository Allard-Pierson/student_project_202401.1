import xml.etree.ElementTree as ET
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import requests
import re

def extract_all_elements_from_table(table):
    elements = {}
    for row in table.find_all('tr'):
        th_element = row.find('th')
        td_element = row.find('td')

        if th_element and td_element:
            element_name = th_element.text.strip()
            element_value = td_element.text.strip()
            elements[element_name] = element_value
    return elements


def process_book(i, books_root):
    new_book = ET.Element('book')
    try:
        response1 = requests.get(f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt")
        if response1.status_code == 200:
            response1_data = BeautifulSoup(response1.text, 'html.parser')
            inhoud = ET.Element('inhoud')
            inhoud.text = re.sub(r'[^ A-Za-z0-9]+', '', str(response1_data)).replace("<","").replace(">","").replace("\n"," ")
            new_book.append(inhoud)
    except Exception as e:
        print(f"Error processing book {i} - response1: {e}")

    try:
        response2 = requests.get(f"https://www.gutenberg.org/ebooks/{i}")
        if response2.status_code == 200:
            response2_data = BeautifulSoup(response2.text, 'html.parser')
            categories = response2_data.find('table', {'class': 'bibrec'})
            elements = extract_all_elements_from_table(categories)

            boek_info = ET.Element('boek_info')
            for element_name, element_value in elements.items():
                if element_name == "Subject":
                    element = ET.Element(element_name)
                    element.text = element_value
                    boek_info.append(element)

            new_book.append(boek_info)
    except Exception as e:
        print(f"Error processing book {i} - response2: {e}")

    if new_book.findall(".//*") or new_book.text:
        # Append the new_book to the common root
        books_root.append(new_book)
    else:
        print(f"Book {i} has no content.")

if __name__ == "__main__":
    # Set the maximum number of threads
    max_threads = 75
    start_index = 0
    end_index = 200
    start_string = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_string = "Project Gutenberg™ eBooks are often created from several printed editions, all of which are confirmed as not protected by copyright in the U.S. unless a copyright notice is included. Thus, we do not necessarily keep eBooks in compliance with any particular paper edition."

    # Initialize the common 'books' root element outside the ThreadPoolExecutor
    books_root = ET.Element('books')

    with ThreadPoolExecutor(max_threads) as executor:
        # Use tqdm to create a progress bar for the loop
        for _ in tqdm(executor.map(process_book, range(start_index, end_index), [books_root]*end_index), total=end_index-start_index, desc="Processing Books", position=0, leave=True):
            pass

    # Write the entire 'books' element to the output file
    with open('./api/dataset/combined_gutenberg_books200.xml', 'w') as output_file:
        output_file.write(ET.tostring(books_root, encoding='utf-8').decode('utf-8'))
