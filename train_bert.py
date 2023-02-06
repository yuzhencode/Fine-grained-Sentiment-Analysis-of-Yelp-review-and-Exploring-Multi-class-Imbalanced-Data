import os
import time

import torch
from loguru import logger
from pytorch_transformers import BertConfig, BertForSequenceClassification
from tqdm import tqdm
import pandas as pd
from data import AllDataset

#禁止并行的意思
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, lossfn, optimizer, dataset, batch_size=32):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for batch, labels in tqdm(generator):
        batch, labels = batch.to(device), labels.to(device)
        optimizer.zero_grad()
        loss, logits = model(batch, labels=labels)
        err = lossfn(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred_labels = torch.argmax(logits, axis=1)
        train_acc += (pred_labels == labels).sum().item()

    train_loss /= len(dataset)
    train_acc /= len(dataset)
    return train_loss, train_acc


def evaluate_one_epoch(model, lossfn, optimizer, dataset, batch_size=32):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.eval()
    loss, acc = 0.0, 0.0
    lab = pd.DataFrame(columns=['label'])
    prelab = pd.DataFrame(columns=['pred'])

    with torch.no_grad():
        for batch, labels in tqdm(generator):
            batch, labels = batch.to(device), labels.to(device)
            logits = model(batch)[0]
            error = lossfn(logits, labels)
            loss += error.item()
            pred_labels = torch.argmax(logits, axis=1)
            acc += (pred_labels == labels).sum().item()
            lab = lab.append(pd.DataFrame(labels.tolist(), columns=['label']))
            prelab = prelab.append(pd.DataFrame(pred_labels.tolist(), columns=['pred']))

    loss /= len(dataset)
    acc /= len(dataset)
    return loss, acc, lab, prelab


def train(
    bert="bert-large-uncased",
    epochs=10,
    batch_size=1,
    save=False,
):
    now = time.strftime("%d-%m-%H-%M", time.localtime())
    trainset = AllDataset("train")
    devset = AllDataset("dev")
    testset = AllDataset("test")

    logfile = open(f"log-{bert}__e{epochs}__{now}.txt", 'w')
    config = BertConfig.from_pretrained(bert)
    config.num_labels = 5
    model = BertForSequenceClassification.from_pretrained(bert, config=config)

    model = model.to(device)
    lossfn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(1, epochs):

        train_loss, train_acc = train_one_epoch(
            model, lossfn, optimizer, trainset, batch_size=batch_size
        )
        val_loss, val_acc, val_lab, val_prelab = evaluate_one_epoch(
            model, lossfn, optimizer, devset, batch_size=batch_size
        )

        test_loss, test_acc, test_lab, test_prelab = evaluate_one_epoch(
            model, lossfn, optimizer, testset, batch_size=batch_size
        )
        now = time.strftime("%d-%m-%H-%M", time.localtime())

        df_out=pd.concat([test_lab, test_prelab], axis=1)
        df_out.to_csv(f"/data/yelp/result/out-{bert}__e{epochs}__{test_acc}__{now}.csv" , sep=',')

        logger.info(f"epoch={epoch}")
        logger.info(
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}"
        )
        logger.info(
            f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, test_acc={test_acc:.3f}"
        )
        logger.info(
            f"test_lab={test_lab}"
        )
        logger.info(
            f"test_prelab={test_prelab}"
        )
        if save:
            label = "fine"
            torch.save(model, f"{bert}__{label}__e{epoch}.pickle")
        logfile.write(f"epoch={epoch}")
        logfile.write(f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}"+"\n")
        logfile.write(f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, test_acc={test_acc:.3f}"+"\n")

    logger.success("Done!")
    logfile.close()


if __name__ == "__main__":
    train(
        bert="bert-base-uncased",
        epochs=10,
        batch_size=32,
        save=False)
