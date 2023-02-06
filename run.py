"""
Run fine-grained sentiment classifier based on chosen method
"""
from typing import Tuple, Any
from classifiers import *
import time


# Path to data
file_path = "data/yelp/"
TRAIN_PATH = file_path + "yelp_train.txt"
DEV_PATH = file_path + "yelp_dev.txt"
TEST_PATH = file_path + "yelp_test.txt"

METHODS = {
    'textblob': {
        'class': "TextBlobClassifier",
        'model': None
    },
    'vader': {
        'class': "VaderClassifier",
        'model': None
    },
     'svm': {
        'class': "SVMClassifier",
        'model': None
    },
    'logistic': {
        'class': "LogisticRegressionClassifier",
        'model': None
    },
    'flair': {
        'class': "FlairClassifier",
        'model': "models/flair/best-model.pt"
    },
    'transformer': {
        'class': "TransformerClassifier",
        'model': "models/transformer",
    },
}


def get_class(method: str, filename: str) -> Any:
    classname = METHODS[method]['class']
    class_ = globals()[classname]
    return class_(filename)


def run_classifier(files: Tuple[str, str, str],
                   method: str,
                   method_class: Base,
                   model_file: str,
                   lower_case: bool) -> None:
    train, dev, test = files
    result = method_class.predict(train, test, lower_case)
    now = time.strftime("%d-%m-%H-%M", time.localtime())
    acc, pre, recall, f1 = method_class.accuracy(result)
    result.to_csv(
        f"{file_path}/result/out-{method}__{acc}__{now}.csv",
        sep=',')


if __name__ == "__main__":
    method_list = [method for method in METHODS.keys()]
    # method is in 'textblob''vader''logistic''svm''flair''transformer'
    method = 'vader'
    model = None
    files = (TRAIN_PATH, DEV_PATH, TEST_PATH)
    lower_case = False
    try:
        if model:
            model_file = model
        else:
            model_file = METHODS[method]['model']
        method_class = get_class(method, model_file)
    except KeyError:
        raise Exception("Please check the method name! {}".format(", ".join(method_list)))

    print("--\nRunning {} classifier on YELP data.".format(method))
    run_classifier(files, method, method_class, model_file, lower_case)
