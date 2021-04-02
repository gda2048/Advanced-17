from collections import Counter

import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

samples_to_test = 2
vectorizer = CountVectorizer()
scaler = StandardScaler()
PERCENT_OF_IMPORTANT_WORDS = 0.8
N = 16184


def _get_transformed_text(titles, fit=True):
    if fit:
        return vectorizer.fit_transform(titles)
    return vectorizer.transform(titles)


def _filter_by_common_words(df, titles):
    counter = Counter([word for word in [line.split() for line in titles.array]])
    counter_top = dict(counter.most_common(int(PERCENT_OF_IMPORTANT_WORDS * len(counter))))
    return df.filter(items=counter_top)


def _get_df(titles, fit):
    transformed_texts = _get_transformed_text(titles, fit)
    df = pd.DataFrame(data=transformed_texts.toarray(), columns=vectorizer.get_feature_names())
    if fit:
        df = _filter_by_common_words(df, titles)
    return df


def _get_result_table(test_data, test_predicted):
    result_table = np.array(['Id', 'Predicted'], dtype=str)
    for idx, id in np.ndenumerate(test_data['Id']):
        result_table = np.append(result_table, [str(int(id)), str(int(test_predicted[idx]))])
    result_table = result_table.reshape(N, 2)
    return result_table


def _get_result_string_from_table(result_table):
    result_string = ""
    for row in result_table:
        result_string += ','.join(map(str, row)) + '\n'
    return result_string


def _get_predicted(test_data):
    test_processed = preprocess_data(test_data, False)
    test_predicted = model.predict(test_processed)
    return test_predicted


def _get_result_string(test_data):
    test_predicted = _get_predicted(test_data)

    result_table = _get_result_table(test_data, test_predicted)
    result_string = _get_result_string_from_table(result_table)
    return result_string


def preprocess_data(data, fit=True):
    titles = data.text.astype(str) + data.title.astype(str)
    df = _get_df(titles, fit)
    for i, title in df.iterrows():
        words_count = len(title.array[0].split())
        df.loc[i] = df.loc[i] / words_count
    if fit:
        scaler.fit(df)
    return preprocessing.normalize(scaler.transform(df))


def generate_submission(model):
    test_data = pd.read_csv("test_without_target.csv")
    with open('submission.csv', 'w') as text_file:
        text_file.write(_get_result_string(test_data))


def train_model():
    train_df = pd.read_csv("train.csv").sample(frac=1)
    sources = train_df[['source']].source.astype(int)
    processed = preprocess_data(train_df)

    model = SVC()
    model.fit(processed[samples_to_test:], sources[samples_to_test:])

    expected, predicted = sources[:samples_to_test].astype(int), model.predict(processed[:samples_to_test])
    print(metrics.classification_report(expected, predicted))
    print(f1_score(expected, predicted, average='macro'))
    return model


model = train_model()
generate_submission(model)
