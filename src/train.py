import datasets
import evaluate
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, logging

def train():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU: ', torch.cuda.get_device_name(0))
    
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device('cpu')
        
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels = 3)
    features = datasets.Features({'text': datasets.Value('string'), 'labels': datasets.ClassLabel(num_classes = 3, names = [-1, 0, 1])})
    dataset = datasets.load_dataset('/home/xic023/DSC180A-Methodology-5/data/out/', features = features)
    train = dataset['train']
    test = dataset['test']
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding = 'max_length', truncation = True)

    tokenized_train = train.map(tokenize_function, batched = True)
    tokenized_test = test.map(tokenize_function, batched = True)
    args = {
        'output_dir': '/home/xic023/DSC180A-Methodology-5/results',
        'evaluation_strategy': 'epoch',
        'num_train_epochs': 1,
        'log_level': 'error',
        'report_to': 'none'
    }
    metric = evaluate.load('accuracy')
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits ,axis = -1)
        return metric.compute(predictions = predictions, references = labels)
    training_args = TrainingArguments(per_device_train_batch_size = 4, **args)
    trainer = Trainer(model = model,
                      args = training_args,
                      train_dataset = tokenized_train,
                      eval_dataset = tokenized_test,
                      tokenizer = tokenizer,
                      compute_metrics = compute_metrics)
    print(trainer.train())
    return trainer
    
