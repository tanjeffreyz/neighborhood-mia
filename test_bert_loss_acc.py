from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, load_metric
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader

# Load the model and tokenizer
model_name = "fabriceyhc/bert-base-uncased-ag_news"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the dataset
dataset = load_dataset("ag_news")

# Take the test or validation split for evaluation
test_dataset = dataset["test"]
# test_dataset = dataset["train"]

# Predicting in batches to avoid memory issues
batch_size = 32
# Use DataLoader for easier batch management
test_dataloader = DataLoader(test_dataset.select(list(range(1000))), batch_size=batch_size, shuffle=False)

predictions = []
true_labels = []
loss = 0
total = 0
for batch in tqdm(test_dataloader):
    tokenized_batch = tokenizer(batch['text'], padding=True, return_tensors='pt', truncation=True)
    tokenized_batch['labels'] = batch['label']
    # Place batch on the same device as the model
    # batch['input_ids']
    # batch = {k: torch.stack(v, dim=0).to(model.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'label', 'token_type_ids']}
    with torch.no_grad():
        outputs = model(**tokenized_batch)
    logits = outputs.logits
    # Convert to probabilities and then to the predicted class
    preds = torch.argmax(logits, dim=-1)
    predictions.extend(preds.cpu().numpy())
    true_labels.extend(batch['label'].cpu().numpy())
    loss += outputs.loss.item() * len(batch['label'])
    total += len(batch['label'])

loss /= total

# Convert predictions to list of predicted labels
predicted_labels = predictions

# Use the datasets load_metric function
accuracy_metric = load_metric("accuracy")
loss_metric = load_metric("glue", "sst2")  # There is no "loss" metric directly, using glue as a proxy

# Calculate accuracy
# true_labels = np.array(test_dataset["label"])
accuracy = accuracy_metric.compute(predictions=predicted_labels, references=true_labels)
print(f'Accuracy: {accuracy}')
print(f'Loss: {loss}')
