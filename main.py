import pandas as pd
import config
import util_ml
import util_stock_price

import matplotlib.pyplot as plt


def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=STOPWORDS,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


def execute_ml():

    raw_data = pd.read_json(config.NEWS_FILE_NAME, lines=True)
    util_ml.perform_EDA(raw_data)
    cleaned_data=util_ml.clean_data(raw_data)
    util_ml.chart_data(cleaned_data)

    # Count Vectorizer
    print('Count Vectorizer')
    x_train, y_train, x_test, y_test = util_ml.count_vectorizer(cleaned_data)
    #util.multi_naive_bayes_model(x_train, y_train, x_test, y_test)
    util_ml.svm_model(x_train, y_train, x_test, y_test)
    #util.random_forest_model(x_train, y_train, x_test, y_test)
    #util.knn_model(x_train, y_train, x_test, y_test)  # KNN Accuracy 0.2518819452742263

    # Hyper Parameter tuning for better model amongst all i.e. SVM
    #util.hyper_tuning_SVM(x_train, y_train)

    # TFIDF
    print('TFIDF)')
    x_train, y_train, x_test, y_test = util_ml.tfidf_vectorizer(cleaned_data)
    #util.multi_naive_bayes_model(x_train, y_train, x_test, y_test)
    util_ml.svm_model(x_train, y_train, x_test, y_test)
    #util.knn_model(x_train, y_train, x_test, y_test)  # KNN Accuracy 0.6452981240291552
    #util.random_forest_model(x_train, y_train, x_test, y_test)

    # Hyper Parameter tuning for better model amongst all i.e. SVM
    #util.hyper_tuning_SVM(x_train, y_train)

def execute_stock_price():
    stock_metadata = util_stock_price.read_json(config.META_FILE_NAME)
    for key in stock_metadata:
        util_stock_price.read_api(config.API_URL, stock_metadata[key], key)

    #util_stock_price.read_csv
    #util_stock_price.read_api(config.API_URL, 'IBM', config.API_KEY)


if __name__ == "__main__":
    execute_stock_price()
else:
    execute_ml()