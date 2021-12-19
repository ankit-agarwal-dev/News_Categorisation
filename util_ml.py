import matplotlib as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import confusion_matrix

def read_json_file(file_name):
    data = pd.read_json(file_name, lines=True)
    return data


def perform_EDA(df):
    df.info()
    df.describe()
    print(df.category.value_counts())
    for col_name in ['headline','short_description','authors']:
        chart_length_data(df[col_name].apply(lambda x: len(str(x).split())),col_name)


def clean_data(uncleaned_df):

    data_cleaning = uncleaned_df.drop(columns=['date','link'])  # Drop Non-useful columns
    data_cleaning.replace("", np.nan, inplace = True)           # Replace all blank values with NaN
    data_cleaning.dropna(axis=0, inplace = True)                # Drop all rows with Null values
    data_cleaning['text'] = data_cleaning['headline'] + '. ' + data_cleaning['short_description'] + '. ' + data_cleaning['authors']
    data_cleaning['text'] = data_cleaning['text'].str.lower()
    data_cleaning['text'].replace(regex=True, inplace=True, to_replace=r'[^0-9a-z ]', value=r'')


    print(data_cleaning)
    # Megre related categories ino one
    merged_categories = merge_categories(data_cleaning, "THE WORLDPOST", "WORLDPOST")
    merged_categories = merge_categories(data_cleaning, "ARTS", "CULTURE & ARTS")
    merged_categories = merge_categories(data_cleaning, "ARTS & CULTURE", "CULTURE & ARTS")
    merged_categories = merge_categories(data_cleaning, "PARENTING", "PARENTS")
    merged_categories = merge_categories(data_cleaning, "STYLE", "STYLE & BEAUTY")
    merged_categories = merge_categories(data_cleaning, "BLACK VOICES", "VOICES")
    merged_categories = merge_categories(data_cleaning, "LATINO VOICES", "VOICES")
    merged_categories = merge_categories(data_cleaning, "QUEER VOICES", "VOICES")
    merged_categories = merge_categories(data_cleaning, "TASTE", "FOOD & DRINK")
    merged_categories = merge_categories(data_cleaning, "GREEN", "ENVIRONMENT")
    merged_categories = merge_categories(data_cleaning, "TECH", "SCIENCE & TECH")
    merged_categories = merge_categories(data_cleaning, "SCIENCE", "SCIENCE & TECH")
    merged_categories = merge_categories(data_cleaning, "COLLEGE", "EDUCATION")
    merged_categories = merge_categories(data_cleaning, "HEALTHY LIVING","HOME & LIVING")
    merged_categories = merge_categories(data_cleaning, "WEDDINGS", "MARRIAGE")
    merged_categories = merge_categories(data_cleaning, "DIVORCE", "MARRIAGE")
    merged_categories = merge_categories(data_cleaning, "BUSINESS", "BUSINESS & FINANCES")
    merged_categories = merge_categories(data_cleaning, "MONEY", "BUSINESS & FINANCES")

    data_wo_stop_word = remove_stop_words(merged_categories)
    print(data_wo_stop_word)
    print(data_wo_stop_word.category.value_counts())
    data_wo_stop_word['text_length'] = data_wo_stop_word.text.apply(lambda i: len(i))
    print(" Text with minimum Length " + str(data_wo_stop_word.text.str.len().min()))
    print(" Duplicate data " + str(data_wo_stop_word.text.duplicated().sum()))
    data_wo_stop_word.drop_duplicates(subset=['text'], inplace=True)
    return data_wo_stop_word


def merge_categories(df, original_category, to_be_replaced_category):
    df['category'].replace(original_category, to_be_replaced_category, inplace=True)
    return df


def chart_data(df):
    sns.set_style('whitegrid')
    sns.countplot(y='category', data=df, palette='RdBu_r',order = df['category'].value_counts().index)
    plt.show()


def chart_length_data(col_data, col_name):
    sns.distplot(col_data)
    plt.title(col_name + ' Number of Words')
    plt.show()


def remove_stop_words(df):
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    df['text_wo_stop_words'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    return df


def word_embedding(df):
    tokenized_sentences = [sentence.split() for sentence in df['text']]
    print(tokenized_sentences)
    X = word2vec(tokenized_sentences, vector_size=100)
    encoder = LabelEncoder()
    Y = encoder.fit_transform(df['merged_category'])
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    return (x_train, y_train, x_test, y_test)

def count_vectorizer(df):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text_wo_stop_words'] )
    encoder = LabelEncoder()
    Y = encoder.fit_transform(df['category'])
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 42)
    return (x_train, y_train, x_test, y_test)


def tfidf_vectorizer(df):
    vectorizer = TfidfVectorizer(min_df=0)
    X = vectorizer.fit_transform(df['text_wo_stop_words'])
    encoder = LabelEncoder()
    Y = encoder.fit_transform(df['category'])

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 42)
    return (x_train, y_train, x_test, y_test)


def multi_naive_bayes_model(X_train, Y_train, X_test, Y_test):
    nb = MultinomialNB()
    nb.fit(X_train, Y_train)
    Y_pred = nb.predict(X_test)
    print("Multi Naive Train Accuracy " + str(nb.score(X_train, Y_train)))
    print("Multi Naive Test Accuracy " + str(nb.score(X_test, Y_test)))
    print(classification_report(Y_test, Y_pred))
    return Y_pred


def knn_model(X_train, Y_train, X_test, Y_test):
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    print("KNN Train Accuracy " + str(knn.score(X_train, Y_train)))
    print("KNN Test Accuracy " + str(knn.score(X_test, Y_test)))
    print(classification_report(Y_test, Y_pred))
    return Y_pred


def svm_model(X_train, Y_train, X_test, Y_test):
    svm_classifier = svm.LinearSVC()
    svm_classifier.fit(X_train, Y_train)
    Y_pred =svm_classifier.predict(X_test)
    print("SVM Train Accuracy " + str(svm_classifier.score(X_train, Y_train)))
    print("SVM Test Accuracy " + str(svm_classifier.score(X_test, Y_test)))
    print(classification_report(Y_test, Y_pred))
    return Y_pred


def random_forest_model(X_train, Y_train, X_test, Y_test):
    random_forest_classifier = RandomForestClassifier(random_state=101).fit(X_train, Y_train)
    Y_pred = random_forest_classifier.predict(X_test)
    print("Random Forest Train Accuracy " + str(random_forest_classifier.score(X_train, Y_train)))
    print("Random Forest Test Accuracy " + str(random_forest_classifier.score(X_test, Y_test)))
    print(classification_report(Y_test, Y_pred))
    return Y_pred

def hyper_tuning_SVM(X_train, Y_train):
    # defining parameter range
    param_grid = {'penalty': ['l2'],
                  'loss': ['hinge','squared_hinge'],
                  'dual': [True],
                  'C' : [0.1, 1, 10, 100, 1000]}

    grid = GridSearchCV(svm.LinearSVC(), param_grid, refit=True, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, Y_train)

    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)


def show_confusion_matrix(test_df, pred_df):
    cf_matrix= confusion_matrix(test_df, pred_df)
    print(cf_matrix)
    ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')

    ax.set_title('Confusion Matrix');
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ');

    ## Display the visualization of the Confusion Matrix.
    plt.show()

