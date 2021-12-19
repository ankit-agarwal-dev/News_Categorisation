import requests
import pandas as pd
import json
import sys
import matplotlib.pyplot as plt
import config


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
        write_csv_file(stock_data_json, symbol)


def write_csv_file(dataset, symbol):
    file_handle=open(symbol + ".csv","w")
    file_handle.write(dataset.text)
    file_handle.close()


def read_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    return data


def prepare_data(df):

    stock_metadata_df = pd.DataFrame(list(df.items()), columns=['stock_name', 'stock_symbol'])

    stock_prices_cf = pd.DataFrame()

    for key in df:
        read_api(config.API_URL, df[key] , config.API_KEY)
        stock_prices = pd.read_csv(str(df[key]) + ".csv", skip_blank_lines=True, header=0,
                                   usecols=['timestamp', 'close'])
        stock_prices['stock_symbol'] = df[key]
        stock_prices_cf = stock_prices_cf.append(stock_prices, ignore_index=True)

    merged_data = pd.merge(stock_prices_cf, stock_metadata_df, on='stock_symbol')
    merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'])

    merged_data.set_index('timestamp', inplace=True)

    return merged_data


def chart_prices(df):

    df.groupby('stock_name')['close'].plot(legend=True)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title("FAANG Adjusted Price Analysis")

    plt.show()