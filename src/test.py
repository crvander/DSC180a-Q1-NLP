import datasets
import yaml
from box import Box
import pandas as pd
import csv
import evaluate
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline, pipeline
import logging

with open('config/test-params.yml', 'r') as file:
    test_config = Box(yaml.full_load(file))
model_path = test_config.model_path
model_name = test_config.model_name
test_path = test_config.test_path
testdata_path = test_config.testdata_path
output_dir = test_config.output_dir
preds_name = test_config.preds_name
preds_detail_name = test_config.preds_detail_name

def test(test_target = 'testdata', test_lines = 3):
    out = []
    if test_target == 'testdata':
        input_path = testdata_path
    if test_target == 'test':
        input_path = test_path
        
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels = 3)
    tokenizer = AutoTokenizer.from_pretrained('{}/{}'.format(model_path, model_name))
    logging.info('initiate testing from {}/{} ...'.format(model_path, model_name))
    logging.info('loading test data from {} ...'.format(input_path))
    testdata = list(pd.read_csv(input_path)['text'].head(test_lines)) # test out the first 50 from test.csv
    logging.info('predicting ...'.format(input_path))
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=1) # return_all_scores will return dict within list 
    
    prediction = pipeline(testdata)
    
    logging.info('saving predictions to {} ...'.format(output_dir))
    myFile = open('{}/{}'.format(output_dir,preds_name), 'w')
    writer = csv.writer(myFile)
    writer.writerow(['label', 'score'])
    for i in prediction:
        writer.writerow(i[0].values())
    myFile.close()        
                
    logging.info('testing done')
    return