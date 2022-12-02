import datasets
import evaluate
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, logging
import logging as lg

def test(trainer, testdata):
    lg.info('initiate testing...')
    testdata = datasets.load_dataset(testdata, features = features)
    testdata = testdata.map(tokenize_function, batched = True)
    out = []
    pred = trainer.predict(testdata['test']).predictions
    for i in pred:
        out.append(int(np.where(i == max(i))[0]) - 1)
    lg.info(str(np.array(out)))
    lg.info('testing done')
    return