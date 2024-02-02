import json
from flask_cors import CORS
from editer.editer import XMLEDITER
from flask import Flask, jsonify, request
from finder.data_finder import DataFinder
from scraper.scraper import SCRAPER
import numpy as np

XMLP = XMLEDITER("./api/dataset/lcc_big_dataset.xml")
XMLP.open_xml()

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST"], "headers": "Content-Type"}})

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Flask API!"})

from flask import stream_with_context, Response

@app.route('/question', methods=['GET'])
def info():
    try:
        params = request.args
        data = params.get('data')
    except:
        return jsonify({"Error":"no input data"})
    #scraper
    scr = SCRAPER()
    links = scr.get_subjects(searchinput=data, model="lcsh-tfidf2-en")
    if links == "error: no data":
        return jsonify({"Error":"no data"})
    lcc_code = []
    model_prediction_score = links
    #scrape the deeper pages also
    for item in links:
        scr = SCRAPER()
        lcc_code.append(scr.get_lcc_code(item))
    #find books and info
    datafinder = DataFinder(XMLP.df)
    found_data_books = datafinder.find_matching_items(lcc_code)
    #return this
    found_data_books = found_data_books[:100]
    # remove null value's
    #found_data_books = found_data_books.fillna('')
    found_data_books = found_data_books.replace(np.nan, 'unknown')
    # for each book, get the info adn put it in json      
    # Convert the DataFrame to a dictionary
    found_data_books_dict = found_data_books.to_dict('records')
    # Convert the dictionary to a JSON object
    found_data_books_json = json.dumps(found_data_books_dict)
    model_prediction_json = json.dumps(model_prediction_score)
    # Assuming model_prediction_json and found_data_books_json are JSON strings
    model_prediction = json.loads(model_prediction_json)
    found_data_books = json.loads(found_data_books_json)

    # Merge the two dictionaries
    response_data = {
        "score": model_prediction,
        "info": found_data_books
    }

    def generate():
        yield json.dumps(response_data)

    return Response(stream_with_context(generate()), mimetype='application/json')

@app.route('/load', methods=['GET'])
def load():
    return jsonify({"message": "Data loaded"})

if __name__ == '__main__':
    app.run(debug=True)

