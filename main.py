import pandas as pd
import config
import util_ml
import util_stock_price
import sys

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

def execute_stock_prices():


    stock_metadata = util_stock_price.read_json(config.META_FILE_NAME)
    prepared_data = util_stock_price.prepare_data(stock_metadata)
    util_stock_price.chart_prices(prepared_data)


if __name__ == "__main__":
    if len(sys.argv) ==1:
        print('Insufficient Arguments. Please pass either "Stock_Prices" or "News_Classification"')
    elif sys.argv[1]=="Stock_Prices":
        execute_stock_prices()
    elif sys.argv[1]=="News_Classification":
        execute_ml()
    else:
        print('Incorrect Arguments. Please pass either "Stock_Prices" or "News_Classification"')