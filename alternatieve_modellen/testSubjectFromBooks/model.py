import logging
from logging import FileHandler
import datetime
import pandas as pd
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import psutil
import traceback

# Set up logging to file
log_file_path = './logs/'  # Define your log file path here
date = datetime.datetime.now()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[FileHandler(log_file_path + str(date) + "-logs.log" , 'a', 'utf-8')])

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes, num_filters):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, max_len, embed_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, max_len, embed_dim]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, bidirectional=True, dropout=0.5):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, max_len, embed_dim]
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        return self.fc(hidden)


def run_model(model_type, train_loader, vocab_size, num_classes, epochs=10):
    # Common parameters for both models
    embed_dim = 100

    # Initialize the chosen model
    if model_type == 'CNN':
        kernel_sizes = [3, 4, 5]
        num_filters = 100
        model = TextCNN(vocab_size, embed_dim, num_classes, kernel_sizes, num_filters)
    elif model_type == 'RNN':
        hidden_dim = 128
        num_layers = 2
        model = TextRNN(vocab_size, embed_dim, hidden_dim, num_layers, num_classes)
    else:
        raise ValueError("Invalid model type. Choose 'CNN' or 'RNN'.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids']
            labels = batch['labels']

            outputs = model(input_ids)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

        # Add evaluation logic here if needed
        # ...

    return model

def text_to_text(data):
    """
    Converts the input text to lowercase.

    Parameters:
    - data (str): The input text to be converted.

    Returns:
    - str: The converted text in lowercase.
    """
    data = data.lower()
    return data

def traditional_machine_learning(X_train, X_test, y_train, y_test, vectorizer, classifier, vectorizer_params={}, classifier_params={}):
    """
    Executes traditional machine learning using vectorization and classification on the provided dataset.

    Parameters:
    - X_train (iterable): Training data set.
    - X_test (iterable): Testing data set.
    - y_train (iterable): Training labels.
    - y_test (iterable): Testing labels.
    - vectorizer (Vectorizer): The vectorizer to use for feature extraction.
    - classifier (Classifier): The classifier for training and prediction.
    - vectorizer_params (dict): Parameters for the vectorizer.
    - classifier_params (dict): Parameters for the classifier.

    Returns:
    - tuple: A tuple containing the training accuracy, testing accuracy, and predictions on the test set.
    """
    vectorizer.set_params(**vectorizer_params)
    X_vec_train = vectorizer.fit_transform(X_train)
    X_vec_test = vectorizer.transform(X_test)

    classifier.set_params(**classifier_params)
    classifier.fit(X_vec_train, y_train)
    memory_usage = psutil.virtual_memory()._asdict()
    logging.info(f"Current memory usage: {memory_usage}")

    y_pred_test = classifier.predict(X_vec_test)
    memory_usage = psutil.virtual_memory()._asdict()
    logging.info(f"Current memory usage: {memory_usage}")

    train_accuracy = classifier.score(X_vec_train, y_train)
    test_accuracy = classifier.score(X_vec_test, y_test)

    return train_accuracy, test_accuracy, y_pred_test


def load_and_process_data(xml_path):
    """
    Loads and processes XML data from the specified path.

    Parameters:
    - xml_path (str): The file path for the XML data.

    Returns:
    - DataFrame: A pandas DataFrame containing the processed data. 
    - None: Returns None if an exception occurs.
    """
    try:
        root = ET.parse(xml_path, parser=ET.XMLParser(encoding='iso-8859-5'))
        data_list = []

        for book_element in root.findall('book'):
            book_data = {}
            inhoud_element = book_element.find('inhoud')
            if inhoud_element is not None:
                book_data['inhoud'] = inhoud_element.text

            boek_info_element = book_element.find('boek_info')
            if boek_info_element is not None:
                for sub_element in boek_info_element:
                    book_data[sub_element.tag] = sub_element.text

            data_list.append(book_data)

        df = pd.DataFrame(data_list)
        for col in df.columns:
            df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)

        df = df.reset_index(drop=True)
        df = df.dropna()
        return df
    except Exception as e:
        logging.error(f"Error in load_and_process_data: {e}")
        logging.error(traceback.format_exc())

class CustomDataset(Dataset):
    """
    Custom Dataset for handling text data suitable for machine learning models, especially for NLP tasks.

    Attributes:
    - texts (iterable): A list of texts.
    - labels (iterable): Corresponding labels for the texts.
    - tokenizer (Tokenizer): Tokenizer for encoding the texts.
    - max_length (int): Maximum length for the encoded text.
    """

    def __init__(self, texts, labels, tokenizer, max_length):
        """
        Initializes the CustomDataset class.

        Parameters:
        - texts (iterable): A list or array of texts to be used as input data.
        - labels (iterable): Corresponding labels for the texts.
        - tokenizer (Tokenizer): A tokenizer instance used to preprocess the text data.
        - max_length (int): The maximum length of the tokenized input.

        Attributes:
        - tokenizer (Tokenizer): Tokenizer for encoding the texts.
        - texts (iterable): A list of texts.
        - labels (iterable): Corresponding labels for the texts.
        - max_length (int): Maximum length for the encoded text.
        """
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
        - int: The number of texts in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        - dict: A dictionary containing encoded inputs, attention masks, and corresponding labels for the specified index.
        """
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }



class LLaMA2Model(nn.Module):
    """
    A hypothetical neural network model resembling LLaMA-2 for classification tasks.

    Parameters:
    - n_classes (int): Number of classes for the classification task.
    """
    def __init__(self, n_classes):
        """
        Initializes the LLaMA2Model class as a subclass of nn.Module. 
        It sets up the LLaMA model and a linear classifier based on the number of classes.

        Parameters:
        - n_classes (int): The number of output classes for the classifier.

        Attributes:
        - LLaMA (AutoModel): Pretrained LLaMA model loaded from 'ProsusAI/finbert'.
        - classifier (nn.Linear): Linear classifier layer, the size depends on the number of classes.
        """
        super(LLaMA2Model, self).__init__()
        self.LLaMA = AutoModel.from_pretrained('ProsusAI/finbert')
        self.classifier = nn.Linear(self.LLaMA.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """
        Defines the forward pass of the LLaMA2Model.

        Parameters:
        - input_ids (tensor): Tensor of input IDs, typically tokenized text.
        - attention_mask (tensor): Tensor representing the attention mask.

        Returns:
        - tensor: Output of the classifier layer, typically logits or scores for each class.
        """
        _, pooled_output = self.LLaMA(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        return self.classifier(pooled_output)

def evaluate_model(model, data_loader):
    """
    Evaluates the given model using the provided data loader and device.

    Parameters:
    - model (torch.nn.Module): The neural network model to evaluate.
    - data_loader (DataLoader): DataLoader containing the dataset for evaluation.
    - device (torch.device): The device on which the model is running.

    Returns:
    - float: The accuracy of the model on the provided dataset.
    """
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for data in data_loader:
            input_ids = data['input_ids']
            attention_mask = data['attention_mask']
            labels = data['labels']
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Main function
def main():
    # Load and preprocess data
    df = load_and_process_data("./api/dataset/combined_gutenberg_books.xml")

    # Encode the 'Subject' column
    label_encoder = LabelEncoder()
    df['Subject'] = label_encoder.fit_transform(df['Subject'])

    # Split the data into training and test sets
    X = df["inhoud"]
    y = df["Subject"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Hypothetical LLaMA-2 Model Training
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    train_dataset = CustomDataset(
        texts=X_train.to_numpy(),
        labels=y_train.to_numpy(),
        tokenizer=tokenizer,
        max_length=128
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = CustomDataset(
        texts=X_test.to_numpy(),
        labels=y_test.to_numpy(),
        tokenizer=tokenizer,
        max_length=128
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Model
    model = LLaMA2Model(n_classes=len(label_encoder.classes_))

    # Training loop for LLaMA-2-like model
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    epochs = 1000  # Adjust as needed

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate during training (optional)
        train_accuracy = evaluate_model(model, train_loader, "cpu")
        val_accuracy = evaluate_model(model, val_loader, "cpu")
        logging.info(f'Epoch {epoch+1}/{epochs}, Train Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}')

    # Save your LLaMA-2-like model
    torch.save(model.state_dict(), './api/models/llama_model.pth')

    # Traditional Machine Learning with CountVectorizer and RandomForestClassifier
    vectorizer = CountVectorizer(preprocessor=text_to_text)
    classifier_rf = RandomForestClassifier(random_state=100, min_samples_split=2, min_samples_leaf=1)
    vectorizer_params = {'ngram_range': (1, 6)}
    classifier_params = {'n_estimators': 1000, 'max_depth': 30}
    train_accuracy_rf, test_accuracy_rf, _ = traditional_machine_learning(
        X_train, X_test, y_train, y_test, vectorizer, classifier_rf, vectorizer_params, classifier_params)
    logging.info(f"Random Forest - Train Accuracy: {train_accuracy_rf}, Test Accuracy: {test_accuracy_rf}")

    # Save RandomForest model
    with open('./api/models/randomforest_model.pkl', 'wb') as file:
        pickle.dump(classifier_rf, file)  

if __name__ == "__main__":
    main()
