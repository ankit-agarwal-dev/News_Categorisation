# Importing System libraries
import sys
import time

# Importing User libraries
import config
import util_ml
import util_stock_price


def execute_ml():
    """
    Description: This is the main program executing various steps for performing a Supervised algorithm.

    :return: None
    """

    print("Starting the News Classification Program...")
    raw_data = util_ml.read_json_file(config.NEWS_FILE_NAME) # Reading Json file for raw labelled data
    util_ml.perform_EDA(raw_data) # Performing EDA on the data set
    cleaned_data=util_ml.clean_data(raw_data) # Cleaning the data
    util_ml.chart_data(cleaned_data) # Charting the cleaned data

    # Count Vectorizer method for textual processing
    print('\nStarting Count Vectorizer Method...')
    # Converting into numerical vector and splitting the data in to test and training sets
    x_train, y_train, x_test, y_test = util_ml.count_vectorizer(cleaned_data)

    # Running Multi Naive Bayes Model
    start_time = time.time()
    y_pred_vect_naive=util_ml.multi_naive_bayes_model(x_train, y_train, x_test, y_test)
    print("Total time taken by Multi Naive Model is :- " + str(time.time() - start_time))
    # Time taken by Multi Naive Bayes model is 0.70 secs and F1 Score is 65%

    # Running LinearSVM Model
    start_time = time.time()
    y_pred_vect_svm =util_ml.svm_model(x_train, y_train, x_test, y_test)
    print("Total time taken by SVM Model is :- " + str(time.time() - start_time))
    # Time taken by Multi SVM model is 47.18 secs and F1 Score is 68%

    # Running Random Forest Model
    start_time = time.time()
    #y_pred_vect_forest=util_ml.random_forest_model(x_train, y_train, x_test, y_test)
    #print("Total time taken by Random Forest Model is :- " + str(time.time() - start_time))
    #  Time taken by Random Forest model is 1608.67 secs and F1 Score is 46%

    # Running KNN Model
    # start_time = time.time()
    # y_pred_vect_knn=util.knn_model(x_train, y_train, x_test, y_test)  # KNN Accuracy 0.2518819452742263
    # print("Total time taken by Random Forest Model is :- " + str(time.time() - start_time))
    # Time taken by KNN model is 1608.67 secs and F1 Score is 46%


    # TFIDF method for textual processing
    print("\nStarting TFIDF Method...")
    # Converting into numerical vector and splitting the data in to test and training sets
    x_train, y_train, x_test, y_test = util_ml.tfidf_vectorizer(cleaned_data)

    # Running Multi Naive Bayes Model
    start_time = time.time()
    y_pred_tfidf_naive =util_ml.multi_naive_bayes_model(x_train, y_train, x_test, y_test)
    print("Total time taken by Multi Naive Model is :- " + str(time.time() - start_time))
    # Time taken by Multi Naive Bayes model is 49.90 secs and F1 Score is 46%

    start_time = time.time()
    y_pred_tfidf_svm =util_ml.svm_model(x_train, y_train, x_test, y_test)
    print("Total time taken by SVM Model is :- " + str(time.time() - start_time))
    # Time taken by Linear SVC model is 14.56 secs and F1 Score is 77%

    #start_time = time.time()
    #y_pred_tfidf_knn=util_ml.knn_model(x_train, y_train, x_test, y_test)  # KNN Accuracy 0.6452981240291552cd
    #print("Total time taken by KNN Model is :- " + str(time.time() - start_time))
    # Time taken by KNN model is 521.65 secs and F1 Score is 61%

    #start_time = time.time()
    #y_pred_tfidf_forest=util_ml.random_forest_model(x_train, y_train, x_test, y_test)
    #print("Total time taken by Random Forest Model is :- " + str(time.time() - start_time))
    #  Time taken by Random Forest model is 1376.05 secs and F1 Score is 68%


    # Hyper Parameter tuning for better model amongst all i.e. SVM
    util_ml.hyper_tuning_SVM(x_train, y_train)

    # Chart Confusion Matrix
    util_ml.show_confusion_matrix(y_test, y_pred_tfidf_svm)
    print(" \n****************** Best Accuracy Achieved using TFIDF and Linear SVM combinaton. Accuracy is 77.66%*****************\n")
    print("News Classification Program successfully completed")


def execute_stock_prices():
    """
    Description:
    Main program for executing Stock prices analysis
    :return:
    """

    print("Starting the Stock Prices Program...")
    # Read metadta for the interested stocks i.e. FAANG group in tis case
    stock_metadata = util_stock_price.read_json(config.META_FILE_NAME)

    # Preparing data for analysis
    prepared_data = util_stock_price.prepare_data(stock_metadata)

    # Charting the price time-series analysis
    util_stock_price.chart_prices(prepared_data)
    print("Stock Prices Program successfully completed")


# Main program for executing the program
if __name__ == "__main__":

    # Checking number of arguments passed and calling the correct program
    if len(sys.argv) ==1:
        print('Insufficient Arguments. Please pass either "Stock_Prices" or "News_Classification"')
    elif sys.argv[1]=="Stock_Prices":
        execute_stock_prices()
    elif sys.argv[1]=="News_Classification":
        execute_ml()
    else:
        print('Incorrect Arguments. Please pass either "Stock_Prices" or "News_Classification"')