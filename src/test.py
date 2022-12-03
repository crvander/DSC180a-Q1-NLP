import datasets
import evaluate
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import logging

def test(trainer, testdata):
    lg.info('initiate testing...')
    testdata = datasets.load_dataset(testdata, features = features)
    testdata = testdata.map(tokenize_function, batched = True)
    out = []
    pred = trainer.predict(testdata['test']).predictions
    for i in pred:
        out.append(int(np.where(i == max(i))[0]) - 1)
    logging.info(str(np.array(out)))
    logging.info('testing done')
    return