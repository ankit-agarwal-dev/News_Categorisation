import requests
import config
import pandas as pd
import json
import sys


def read_api(URL, symbol, key):


def read_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    return data


    api_endpoint = URL + str(symbol) +"&" + "apikey="+ config.API_KEY
    stock_data_json = requests.get(api_endpoint)
    print(stock_data_json.status_code)
    print(stock_data_json[0])
    try:
        for key in stock_data_json.json():
            if key == 'Note':
                print("API key limit Exceeded. Limit is 5 calls per minute or 500 calls per day")
                sys.exit(1)
    except:
        print(repr(stock_data_json))
        print(sys.exc_info())
        print(stock_data_json.text)

    write_csv_file(stock_data_json, symbol)

def write_csv_file(dataset, symbol):
    file_handle=open(symbol + ".csv","w")
    file_handle.write(dataset.text)
    file_handle.close()