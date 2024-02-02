import subprocess
import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote

class SCRAPER:
    def __init__(self):
        pass
    
    def get_subjects(self, searchinput, model="lcsh-omikuji-parabel-en", cwd="./api/annif"):
        '''
        This function takes input text, an annif model and optionally the path of the working directory,
        it returns the given subjects of the model in a dictionary.
        '''
        data = {}  # Dictionary for the values
        command = f"echo {searchinput} | annif suggest {model}" #Command for getting subjects with input text
        result = subprocess.run(command, stdout=subprocess.PIPE, shell=True, text=True, cwd=cwd)  # Saves output of command in text
        output = result.stdout.strip()  # Stript redundant information from result
        data_lines = output.split('\n')  # Split the string into multiple lines
        if data_lines[0] == "":
            return "error: no data"
        
        for line in data_lines:
            tokens = line.split()  # Split line into tokens
            key = tokens[0]  # Subject link as key
            subject = ' '.join(tokens[1:-1])  # Join is used to handle subject with spaces
            score = float(tokens[-1])  # Accuracy score
            data[key] = {'subject': subject, 'score': score}

        print(f"annif found: {data}")
        return data

    def get_lcc_code(self, uri):
        '''
        This function takes a uri, it return the LC classification code if possible
        '''
        response = requests.get(unquote(uri.strip('<>')))  # Reads URI.
        # Check if the request was successful (code == 200)
        if response.status_code == 200:
            # Parse the HTML of the page
            soup = BeautifulSoup(response.text, 'html.parser')
            # Find <li> element with property="madsrdf:classification", this contains the LCC code.
            lcc_li = soup.find('li', {'property': 'madsrdf:classification'})
            # Extract the text content inside the <li> element
            lcc_code = lcc_li.text.strip() if lcc_li else None
            # Print the result
            return lcc_code
        else:
            print("Unable to reach page, status code:", response.status_code)
