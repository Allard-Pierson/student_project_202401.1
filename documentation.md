# Code Documentation

## API

### 1. Home

**Route:** `/`

**Method:** GET

**Description:** Returns a welcome message indicating successful connection to the Flask API.

#### Request

```plaintext
GET /
```

#### Response

```json
{
  "message": "Welcome to the Flask API!"
}
```

### 2. Information Retrieval

**Route:** `/question`

**Method:** GET

**Description:** Retrieves book information based on user queries. It uses a web scraper to search for relevant subjects, extracts LCC (Library of Congress Classification) codes, and matches them with available book data.

#### Request

```plaintext
GET /question?data=<search_query>
```

**Parameters:**
- `data` (string): User input for book search.

#### Response

```json
{
  "info": [
    {
      // Book information
    },
    // Additional book entries
  ]
}
```

### 3. Data Loading

**Route:** `/load`

**Method:** GET

**Description:** Returns a message indicating that the data has been loaded.

#### Request

```plaintext
GET /load
```

#### Response

```json
{
  "message": "Data loaded"
}
```

# Dependencies

- Flask: Micro web framework for Python.
- Flask-CORS: Extension for handling Cross-Origin Resource Sharing.
- XMLEDITER: Module for parsing and handling XML data.
- DataFinder: Module for searching and extracting book data.
- SCRAPER: Module for web scraping to find subjects and LCC codes.
- NumPy: Library for numerical operations in Python.

## Annif
### Overview
The `Annif` folder contains functions and notebooks for creating the dataset for Annif and training Annif projects. For more detailed documentation, please visit the associated notebooks. [data_processing.ipynb](api/annif/data_processing.ipynb)

### Methods

1. __ubxml_to_titles__
```python
def ubxml_to_titles(xml_files, output_csv):
    """
    This function takes a list of book MARC XML files (from UB) and an output_csv target file
    and returns the titles & subjects into a single CSV.

    Parameters:
    - xml_files: Path to the XML files
    - output_csv: Path to the output .csv file
    """
```

2. __snxml_to_titles__
```python
def snxml_to_titles(xml_file, output_csv):
    """
    This function takes a MARC XML file (from Springer Nature) and an output_csv target file
    and returns the titles & subjects into a single CSV.

    Parameters:
    - xml_file: Path to the XML file
    - output_csv: Path to the output .csv file
    """
```
3. __ubxml_to_summaries__
```python
def ubxml_to_summaries(xml_files, output_csv):
    """
    This function takes a list of book MARC XML files and an output_csv target file
    and returns the summaries & subjects into a single CSV.

    Parameters:
    - xml_files: Path to the XML files
    - output_csv: Path to the output .csv file
    """
```

4. __snxml_to_summaries__
```python
def snxml_to_titles(xml_file, output_csv):
    """
    This function takes a MARC XML file (from Springer Nature) and an output_csv target file
    and returns the summaries & subjects into a single CSV.

    Parameters:
    - xml_file: Path to the XML file
    - output_csv: Path to the output .csv file
    """
```

5. __clean_dataset__
```python
def clean_dataset(dataset, clean_dataset):
  """
    Takes a raw dataset, removes duplicates and unnecessary symbols, removes non-english text, 
    returns new dataset and examples during cleaning.

    Parameters:
    - xml_file: Path to the original dataset
    - output_csv: Path to the output .csv file

    Returns:
    - original data: pd.Dataframe head of original data
    - clean_data_step1: pd.Dataframe head of data during cleaning
    - clean_data_step2: pd.Dataframe head of data after cleaning

  """
```
6. __detect_language__
```python
def detect_language(text):
  """
  Takes a text and returns the classified language and the confidence score

  Parameters:
  - text: input text

  Returns:
  - lang: classified language
  - confidence: confidence score of correct language.
  """
```

7. __shift_subjects__
```python
def shift_subjects(input_tsv, output_tsv):
  """
    This function takes an input- and output .tsv filepath.
    It shifts all the subjects to the left
    Returns the altered dataset and before and after of the first few entries.

    Parameters:
    - input_tsv: Path to the input .tsv file
    - output_csv: Path to the output .tsv file

    Returns:
    - original_data: pd.Dataframe head of original data
    - new_data: pd.Dataframe head of new data
  """
```

### Example Usage

For example usage please view the aforementioned notebooks.

## XMLEDITER Class

### Overview

The `XMLEDITER` class is designed for parsing and handling XML data related to book information. It includes methods for opening an XML file, removing NaN values, and filtering rows without Library of Congress Classification (LCC) classification.

### Constructor

```python
class XMLEDITER:
    def __init__(self, xml_file: str):
        """
        Initializes the XMLEDITER class.

        Parameters:
        - xml_file (str): Path to the XML file.

        Raises:
        - FileNotFoundError: If the specified XML file is not found.
        """
```

### Methods

1. **open_xml**

   ```python
   def open_xml(self) -> pd.DataFrame:
       """
       Reads and parses the XML file, storing the data in a pandas DataFrame.

       Returns:
       - pd.DataFrame: The DataFrame containing the XML data.
       """
   ```

2. **remove_na**

   ```python
   def remove_na(self) -> pd.DataFrame:
       """
       Fills NaN values in the DataFrame with empty strings.

       Returns:
       - pd.DataFrame: The DataFrame with NaN values replaced.
       """
   ```

3. **dropnolcc**

   ```python
   def dropnolcc(self) -> pd.DataFrame:
       """
       Removes rows without a Library of Congress Classification (LCC) classification.

       Returns:
       - pd.DataFrame: The DataFrame with rows having LCC classification.
       """
   ```

### Example Usage

```python
# Example instantiation
xml_parser = XMLEDITER("./path/to/your/xml/file.xml")

# Example usage of methods
xml_parser.open_xml()
xml_parser.remove_na()
xml_parser.dropnolcc()
```

## DataFinder Class

### Overview

The `DataFinder` class is designed to search and filter data based on specific criteria. It includes a method for finding items that match a given set of values.

### Constructor

```python
class DataFinder:
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the DataFinder class.

        Parameters:
        - data (pd.DataFrame): The DataFrame containing the data to be searched.
        """
```

### Methods

1. **find_matching_items**

   ```python
   def find_matching_items(self, value: list) -> pd.DataFrame:
       """
       Finds and returns items in the DataFrame where the specified column matches any value in the given list.

       Parameters:
       - value (list): List of values to search for in the specified column.

       Returns:
       - pd.DataFrame: Subset of the original DataFrame containing matching items.
       """
   ```

### Example Usage

```python
# Example instantiation
data_finder = DataFinder(your_dataframe)

# Example usage of methods
matching_items = data_finder.find_matching_items(['value1', 'value2', 'value3'])
print(matching_items)

```

## SCRAPER Class

### Overview

The `SCRAPER` class is responsible for scraping information related to subjects and LCC codes from external sources. It includes methods for retrieving subjects based on input text and obtaining LCC codes from URIs.

### Constructor
```python

class SCRAPER:
    def __init__(self):
        """
        Initializes the SCRAPER class.
        """
```

### Methods

1. **get_subjects**

   ```python
   def get_subjects(self, searchinput: str, model: str = "lcsh-tfidf-en", cwd: str = "./api/annif") -> dict or str:
       """
       Retrieves subjects based on the input text using the specified Annif model.

       Parameters:
       - searchinput (str): The input text for subject prediction.
       - model (str): The Annif model to use (default is "lcsh-tfidf-en").
       - cwd (str): The path to the working directory (default is "./api/annif").

       Returns:
       - dict or str: A dictionary containing subjects and their scores, or an error message if no data is available.
       """
   ```

2. **get_lcc_code**

   ```python
   def get_lcc_code(self, uri: str) -> str or None:
       """
       Retrieves the LC classification code from a given URI.

       Parameters:
       - uri (str): The URI from which to extract the LCC code.

       Returns:
       - str or None: The extracted LCC code, or None if unsuccessful.
       """
   ```

### Example Usage

```python
# Example instantiation
scraper = SCRAPER()

# Example usage of methods
subjects = scraper.get_subjects("input_text", model="lcsh-tfidf-en")
print(subjects)

uri = "example_uri"
lcc_code = scraper.get_lcc_code(uri)
print(lcc_code)
```
