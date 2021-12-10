import requests
import config
import pandas as pd
import json
import sys


def read_api(URL, symbol, key):
    limit_exceeded = 0
    api_endpoint = URL + str(symbol) +"&" + "apikey="+ key
    stock_data_json = requests.get(api_endpoint)
    try:
        for key in stock_data_json.json():
            if key == 'Note':
                limit_exceeded = 1
    except:
        pass;
    if limit_exceeded == 1:
        print("API key limit Exceeded. Limit is 5 calls per minute or 500 calls per day")
        return sys.exit()
    else:
        print("Exiting Program")
        write_csv_file(stock_data_json, symbol)


def write_csv_file(dataset, symbol):
    file_handle=open(symbol + ".csv","w")
    file_handle.write(dataset.text)
    file_handle.close()


def read_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    return data
