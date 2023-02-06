#!/usr/bin/env python3
from typing import Tuple
from pathlib import Path


def trainer(file_path: Path,
            filenames: Tuple[str, str, str],
            checkpoint: str,
            n_epochs: int) -> None:

    # pip install flair allennlp
    from flair.datasets import ClassificationCorpus
    from flair.embeddings import FlairEmbeddings, DocumentRNNEmbeddings
    from flair.models import TextClassifier
    from flair.trainers import ModelTrainer

    from flair.embeddings import WordEmbeddings
    stacked_embedding = WordEmbeddings('glove')

    train, dev, test = filenames
    corpus = ClassificationCorpus(
        file_path,
        train_file=train,
        dev_file=dev,
        test_file=test,
    )
    label_dict = corpus.make_label_dictionary(label_type='class')

    word_embeddings = list(filter(None, [
        stacked_embedding,
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]))
    document_embeddings = DocumentRNNEmbeddings(
        word_embeddings,
        hidden_size=512,
        reproject_words=True,
        reproject_words_dimension=256,
    )
    classifier = TextClassifier(
        document_embeddings,
        label_dictionary=label_dict,
        multi_label=False,
        label_type='class'
    )

    if not checkpoint:
        trainer = ModelTrainer(classifier, corpus)
    else:
        checkpoint = classifier.load_checkpoint(Path(checkpoint))
        trainer = ModelTrainer.load_from_checkpoint(checkpoint, corpus)

    trainer.train(
        file_path,
        learning_rate = 0.1,
        max_epochs=n_epochs,
        checkpoint=True
    )


if __name__ == "__main__":

    path = "data/yelp"
    train = "yelp_train_small.txt"
    dev = "yelp_dev.txt"
    test = "yelp_test.txt"
    filepath = Path('./') / path
    filenames = (train, dev, test)
    epochs = 25
    trainer(filepath, filenames, checkpoint = None, n_epochs=epochs)
