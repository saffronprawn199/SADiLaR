#!/usr/bin/env python
import os
import logging
import multiprocessing
import pandas as pd
import gensim
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec
from itertools import islice
import argparse

os.system("taskset -p 0xff %d" % os.getpid())
logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

cores = multiprocessing.cpu_count()
"""Use this when you have to little RAM to store all your data"""

# Download popular NLTK dataset
nltk.download("popular")


def load_data(data_file):
    df = pd.read_csv(data_file)
    df = df[pd.notnull(df["Sentence"])]
    df = df[["Tag", "Sentence"]]
    df["Sentence"] = df["Sentence"].apply(clean_text)
    # print_df(df)
    return df


def print_df(df):
    print(df.head())


def clean_text(text):
    """
    text: a string

    return: modified initial string
    """
    # define regexes
    replace_by_space_re = re.compile("[/(){}[]|@,;]")
    bad_symbols_re = re.compile("[^0-9a-z #+_]")
    stopwords_set = set(stopwords.words("english"))
    text = text.lower()  # lowercase text
    text = replace_by_space_re.sub(
        " ", text
    )  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = bad_symbols_re.sub(
        "", text
    )  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join(
        word for word in text.split() if word not in stopwords_set
    )  # delete stopwords from text
    return text


def word_averaging(wv, words):
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.index_to_key:
            mean.append(wv.vectors[wv.key_to_index[word]])
            all_words.add(wv.key_to_index[word])

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        return np.zeros(
            wv.vector_size,
        )

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)

    return mean


def word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list])


def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language="english"):
        for word in nltk.word_tokenize(sent, language="english"):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens


def print_report(y_pred, y_test, y_pred_prob, my_tags, name, output_file):
    """
    make dir to save report to
    create report
    print report to new file or append to previous file
    :param y_pred: prediction values
    :param y_test: test values
    :param my_tags: list of user specified classification tags
    :param name: string: model name
    :param output_file: output file name
    """
    classifier_path = "./classification/"
    os.makedirs(classifier_path, exist_ok=True)

    file = open(classifier_path + output_file, "a")
    file.write(name + ":\n")
    file.write("accuracy %s" % accuracy_score(y_pred, y_test) + "\n")

    # label_encoder = LabelEncoder()
    # y_test_encoded = label_encoder.transform(y_test)

    file.write(
        "AUC %s"
        % roc_auc_score(y_test, y_pred_prob, multi_class="ovr", average="micro")
        + "\n"
    )

    file.write(classification_report(y_test, y_pred, target_names=my_tags) + "\n")
    file.close()


def train_baseline(df):
    """
    split data into testing and training
    create classification pipeline
    fit classification model
    :param df: pandas dataset
    :return: prediction and test values
    """
    x = df.Sentence
    y = df.Tag
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    logreg = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", LogisticRegression(n_jobs=1, C=1e5, multi_class="ovr")),
        ]
    )
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    y_pred_prob = logreg.predict_proba(x_test)

    return y_pred, y_test, y_pred_prob


def train_w2v_classifier(model, df):
    """
    load pre-trained embedding model
    split data into train and test split & tokenize
    normalise vectors
    define classification model, fit and predict
    :param model: pre-trained embedding model
    :param df: pandas dataframe
    :return: prediction and test values
    """
    #  Test with gensim's pretrained embeddings=========================================================================
    #  wv = gensim.models.KeyedVectors.load_word2vec_format(model, binary=True)
    #  wv.init_sims(replace=True)
    #  =================================================================================================================
    #  To use pretrained wordvectors====================================================================================
    wordvec_model = gensim.models.Word2Vec.load(model)
    # wordvec_model.init_sims(replace=True)
    wv = wordvec_model.wv
    # del wordvec_model
    #  =================================================================================================================
    list(islice(wv.index_to_key, 13030, 13050))

    train, test = train_test_split(df, test_size=0.3, random_state=42)

    test_tokenized = test.apply(
        lambda r: w2v_tokenize_text(r["Sentence"]), axis=1
    ).values
    train_tokenized = train.apply(
        lambda r: w2v_tokenize_text(r["Sentence"]), axis=1
    ).values

    x_train_word_average = word_averaging_list(wv, train_tokenized)
    x_test_word_average = word_averaging_list(wv, test_tokenized)

    logreg = LogisticRegression(n_jobs=-1, C=1e5, multi_class="ovr")
    logreg = logreg.fit(x_train_word_average, train["Tag"])
    y_pred = logreg.predict(x_test_word_average)
    y_pred_prob = logreg.predict_proba(x_test_word_average)

    return y_pred, test.Tag, y_pred_prob


def get_w2v_classifier(model, df, my_tags, output_file):
    """
    Call classification function
    Call print function
    :param model: pretrained embedding model
    :param df: pandas dataset
    :param my_tags: list of user specified classification tags
    :param output_file: output file name
    """
    y_pred, y_test, y_pred_prob = train_w2v_classifier(model, df)
    name = "Classification model using W2V"
    print_report(y_pred, y_test, y_pred_prob, my_tags, name, output_file)
    print("Classification done")


def get_baseline(df, my_tags, output_file):
    """
    Call classification function
    Call print function
    :param df: pandas dataset
    :param my_tags: list of user specified classification tags
    :param output_file: output file name
    """
    y_pred, y_test, y_pred_prob = train_baseline(df)
    name = "Baseline classification model"
    print_report(y_pred, y_test, y_pred_prob, my_tags, name, output_file)
    print("Classification done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify pre-trained word embedding models \
                                                        and baseline classification model."
    )
    requiredNamed = parser.add_argument_group("Required named arguments")

    requiredNamed.add_argument(
        "--inFilePath",
        action="store",
        metavar="string",
        type=str,
        dest="inFilePath",
        help="defines the input csv file path",
        required=True,
    )

    requiredNamed.add_argument(
        "--inModelPath",
        action="store",
        metavar="string",
        type=str,
        dest="inModelPath",
        help="specifies word embedding input file name",
        required=True,
    )

    requiredNamed.add_argument(
        "--outFileName",
        action="store",
        metavar="string",
        type=str,
        dest="outFileName",
        help="specifies classification model output file name",
        required=True,
    )
    parser.add_argument(
        "--BL",
        metavar="bool",
        type=bool,
        default=False,
        help="specifies if the baseline classification model should be trained",
    )
    parser.add_argument(
        "-tags",
        "--tag",
        type=str,
        default="news,sport,weather,advertisement,traffic",
        help="delimited list input that specifies the classes",
    )

    args = parser.parse_args()
    data = args.inFilePath
    out_file = args.outFileName
    pretrained_model = args.inModelPath
    tags = [item for item in args.tag.split(",")]
    dataframe = load_data(data)
    if args.BL == 1:
        get_baseline(dataframe, tags, out_file)
        get_w2v_classifier(pretrained_model, dataframe, tags, out_file)
    else:
        get_w2v_classifier(pretrained_model, dataframe, tags, out_file)
