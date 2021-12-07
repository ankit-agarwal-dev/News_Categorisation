import pandas as pd
import config
import util


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


def execute():
    
    raw_data = pd.read_json(config.file_name, lines=True)
    util.perform_EDA(raw_data)
    cleaned_data=util.clean_data(raw_data)
    util.chart_data(cleaned_data)
    cleaned_data['text_length'] = cleaned_data.text.apply(lambda i: len(i))
    print(" Text with minimum Length " + str(cleaned_data.text.str.len().min()))

    print(cleaned_data.shape)
    print(" Duplicate data " + str(cleaned_data.headline.duplicated().sum()))

    cleaned_data.drop_duplicates(subset=['text'], inplace=True)
    print(cleaned_data.shape)


    # Count Vectorizer
    print('Count Vectorizer')
    x_train, y_train, x_test, y_test = util.count_vectorizer(cleaned_data)
    #util.multi_naive_bayes_model(x_train, y_train, x_test, y_test)
    util.svm_model(x_train, y_train, x_test, y_test)
    #util.random_forest_model(x_train, y_train, x_test, y_test)
    #util.knn_model(x_train, y_train, x_test, y_test)  # KNN Accuracy 0.2518819452742263

    # Hyper Parameter tuning for better model amongst all i.e. SVM
    #util.hyper_tuning_SVM(x_train, y_train)

    # TFIDF
    print('TFIDF)')
    x_train, y_train, x_test, y_test = util.tfidf_vectorizer(cleaned_data)
    #util.multi_naive_bayes_model(x_train, y_train, x_test, y_test)
    util.svm_model(x_train, y_train, x_test, y_test)
    #util.knn_model(x_train, y_train, x_test, y_test)  # KNN Accuracy 0.6452981240291552
    #util.random_forest_model(x_train, y_train, x_test, y_test)

    # Hyper Parameter tuning for better model amongst all i.e. SVM
    #util.hyper_tuning_SVM(x_train, y_train)




if __name__ == "__main__":
    execute()
else:
    execute()