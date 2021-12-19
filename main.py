import config
import util_ml
import util_stock_price
import sys
import time


def execute_ml():

    raw_data = util_ml.read_json_file(config.NEWS_FILE_NAME)
    util_ml.perform_EDA(raw_data)
    cleaned_data=util_ml.clean_data(raw_data)
    util_ml.chart_data(cleaned_data)

    # Count Vectorizer
    print('Count Vectorizer')
    x_train, y_train, x_test, y_test = util_ml.count_vectorizer(cleaned_data)
    start_time = time.time()
    util_ml.multi_naive_bayes_model(x_train, y_train, x_test, y_test)
    print("Total time taken by Multi Naive Model is :- " + str(time.time() - start_time))  # Time taken is 0.70 secs and F1 Score is 65%
    start_time = time.time()
    util_ml.svm_model(x_train, y_train, x_test, y_test)
    print("Total time taken by SVM Model is :- " + str(time.time() - start_time))  # Time taken is 47.18 secs and F1 Score is 68%
    start_time = time.time()
    util_ml.random_forest_model(x_train, y_train, x_test, y_test)
    print("Total time taken by Random Forest Model is :- " + str(time.time() - start_time))  # Time taken is 1608.67 secs and F1 Score is 46%

    #util.knn_model(x_train, y_train, x_test, y_test)  # KNN Accuracy 0.2518819452742263

    # Hyper Parameter tuning for better model amongst all i.e. SVM
    #util.hyper_tuning_SVM(x_train, y_train)

    # TFIDF
    print('TFIDF)')
    x_train, y_train, x_test, y_test = util_ml.tfidf_vectorizer(cleaned_data)
    start_time = time.time()
    util_ml.multi_naive_bayes_model(x_train, y_train, x_test, y_test)
    print("Total time taken by Multi Naive Model is :- " + str(time.time() - start_time)) # Time taken is 49.90 secs and F1 Score is 46%
    start_time = time.time()
    util_ml.svm_model(x_train, y_train, x_test, y_test)
    print("Total time taken by SVM Model is :- " + str(time.time() - start_time))  # Time taken is 14.56 secs and F1 Score is 77%
    start_time = time.time()
    util_ml.knn_model(x_train, y_train, x_test, y_test)  # KNN Accuracy 0.6452981240291552cd
    print("Total time taken by KNN Model is :- " + str(time.time() - start_time)) # Time taken is 521.65 secs and F1 Score is 61%
    start_time = time.time()
    util_ml.random_forest_model(x_train, y_train, x_test, y_test)
    print("Total time taken by Random Forest Model is :- " + str(time.time() - start_time)) # Time taken is 1376.05 secs and F1 Score is 68%
    # Hyper Parameter tuning for better model amongst all i.e. SVM
    start_time = time.time()
    util_ml.hyper_tuning_SVM(x_train, y_train)

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