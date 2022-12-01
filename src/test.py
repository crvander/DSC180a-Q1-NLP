import datasets
import evaluate
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, logging
from src.models.train import train

def test(trainer, testdata):
    testdata = datasets.load_dataset(testdata, features = features)
    testdata = testdata.map(tokenize_function, batched = True)
    out = []
    pred = trainer.predict(testdata['test']).predictions
    for i in pred:
        out.append(int(np.where(i == max(i))[0]) - 1)
    return np.array(out)