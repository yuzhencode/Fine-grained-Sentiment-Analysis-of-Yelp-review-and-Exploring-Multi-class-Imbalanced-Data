import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


class Base:
    def __init__(self) -> None:
        pass

    def read_data(self, fname: str, lower_case: bool=False,
                  colnames=['label', 'text']):
        df = pd.read_csv(fname, sep='\t', header=None, names=colnames)
        df['label'] = df['label'].str.replace('__label__', '')
        df['label'] = df['label'].astype(int).astype('category')
        if lower_case:
            df['text'] = df['text'].str.lower()
        return df

    def accuracy(self, df: pd.DataFrame):
        acc = accuracy_score(df['label'], df['pred'])*100
        pre = precision_score(df['label'], df['pred'], average='macro')*100
        recall = recall_score(df['label'], df['pred'], average='macro')*100
        f1 = f1_score(df['label'], df['pred'], average='macro')*100
        print("Accuracy: {:.3f}\nMacro Precision_score: {:.3f}\nMacro Recall_score: {:.3f}\nMacro F1-score: {:.3f}".format(acc, pre, recall, f1))
        return acc, pre, recall, f1

class TextBlobClassifier(Base):
    """
    https://textblob.readthedocs.io/en/dev/
    """
    def __init__(self, model_file: str=None):
        super().__init__()

    def score(self, text: str):
        # pip install textblob
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity

    def predict(self, train_file: None, test_file: str, lower_case: bool):
        df = self.read_data(test_file, lower_case)
        df['score'] = df['text'].apply(self.score)
        df['pred'] = pd.cut(df['score'],
                            bins=5,
                            labels=[1, 2, 3, 4, 5])
        df = df.drop('score', axis=1)
        return df


class VaderClassifier(Base):
    """
    https://www.nltk.org/_modules/nltk/sentiment/vader.html
    """
    def __init__(self, model_file: str=None):
        super().__init__()
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

    def score(self, text: str):
        return self.vader.polarity_scores(text)['compound']

    def predict(self, train_file: None, test_file: str, lower_case: bool):
        df = self.read_data(test_file, lower_case)
        df['score'] = df['text'].apply(self.score)
        df['pred'] = pd.cut(df['score'],
                            bins=5,
                            labels=[1, 2, 3, 4, 5])
        df = df.drop('score', axis=1)
        return df


class SVMClassifier(Base):
    def __init__(self, model_file: str=None):
        super().__init__()
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline
        self.pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(
                    loss='hinge',
                    penalty='l2',
                    alpha=1e-3,
                    random_state=42,
                    max_iter=100,
                    learning_rate='optimal',
                    tol=None,
                )),
            ]
        )

    def predict(self, train_file: str, test_file: str, lower_case: bool):
        train_df = self.read_data(train_file, lower_case)
        learner = self.pipeline.fit(train_df['text'], train_df['label'])
        test_df = self.read_data(test_file, lower_case)
        test_df['pred'] = learner.predict(test_df['text'])
        return test_df


class LogisticRegressionClassifier(Base):
    def __init__(self, model_file: str=None):
        super().__init__()
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        self.pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(
                    solver='newton-cg',
                    multi_class='multinomial',
                    random_state=42,
                    max_iter=100,
                )),
            ]
        )

    def predict(self, train_file: str, test_file: str, lower_case: bool):
        train_df = self.read_data(train_file, lower_case)
        learner = self.pipeline.fit(train_df['text'], train_df['label'])
        test_df = self.read_data(test_file, lower_case)
        test_df['pred'] = learner.predict(test_df['text'])
        return test_df


class FlairClassifier(Base):
    def __init__(self, model_file: str=None):
        super().__init__()
        "Use the latest version of Flair NLP from their GitHub repo!"
        # pip install flair
        from flair.models import TextClassifier
        try:
            self.model = TextClassifier.load(model_file)
        except ValueError:
            raise Exception("Please specify a valid trained Flair PyTorch model file (.pt extension)'{}'."
                            .format(model_file))

    def score(self, text: str):
        from flair.data import Sentence
        doc = Sentence(text)
        self.model.predict(doc)
        pred = int(doc.labels[0].value)
        return pred

    def predict(self, train_file: None, test_file: str, lower_case: bool):
        from tqdm import tqdm
        tqdm.pandas()
        df = self.read_data(test_file, lower_case)
        df['pred'] = df['text'].progress_apply(self.score)
        return df


class TransformerClassifier(Base):
    def __init__(self, model_path: str = None):
        super().__init__()
        "Requires the BertTokenizer from pytorch_transformers"
        import os
        import torch
        from pytorch_transformers import BertTokenizer, cached_path
        from model_transformer import TransformerWithClfHeadAndAdapters
        try:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.config = torch.load(cached_path(os.path.join(model_path, "model_training_args.bin")))
            self.model = TransformerWithClfHeadAndAdapters(self.config["config"],
                                                           self.config["config_ft"]).to(self.device)
            state_dict = torch.load(cached_path(os.path.join(model_path, "model_weights.pth")),
                                    map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        except:
            raise Exception("Require a valid transformer model file ({0}/model_weights.pth) "
                            "and its config file ({0}/model_training_args.bin)."
                            .format(model_path))

    def encode(self, inputs):
        return list(self.tokenizer.convert_tokens_to_ids(o) for o in inputs)

    def score(self, text: str):
        "Return an integer value of predicted class from the transformer model."
        import torch
        import torch.nn.functional as F

        self.model.eval()  # Disable dropout
        clf_token = self.tokenizer.vocab['[CLS]']  # classifier token
        pad_token = self.tokenizer.vocab['[PAD]']  # pad token
        max_length = self.config['config'].num_max_positions  # Max length from trained model
        inputs = self.tokenizer.tokenize(text)
        if len(inputs) >= max_length:
            inputs = inputs[:max_length - 1]
        ids = self.encode(inputs) + [clf_token]

        with torch.no_grad():  # Disable backprop
            tensor = torch.tensor(ids, dtype=torch.long).to(self.device)
            tensor = tensor.reshape(1, -1)
            tensor_in = tensor.transpose(0, 1).contiguous()  # to shape [seq length, 1]
            logits = self.model(tensor_in,
                                clf_tokens_mask=(tensor_in == clf_token),
                                padding_mask=(tensor == pad_token))
        val, _ = torch.max(logits, 0)
        val = F.softmax(val, dim=0).detach().cpu().numpy()
        # To train the transformer in PyTorch we zero-indexed the labels.
        # Now we increment the predicted label by 1 to match with those from other classifiers.
        pred = int(val.argmax()) + 1
        return pred

    def predict(self, train_file: None, test_file: str, lower_case: bool):
        from tqdm import tqdm
        tqdm.pandas()
        df = self.read_data(test_file, lower_case)
        df['pred'] = df['text'].progress_apply(self.score)
        return df

