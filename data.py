import pandas as pd
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast, Trainer, TrainingArguments, pipeline
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# === 1. Load your CSV dataset ===
file_path = "path/to/your/data.csv"  # <-- Replace this with your CSV file path
df = pd.read_csv(file_path)

# === 2. Convert dataframe to HuggingFace Dataset ===
dataset = Dataset.from_pandas(df)

# === 3. Load tokenizer and model ===
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

# === 4. Tokenize the dataset ===
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# === 5. Split dataset (80% train, 20% eval) ===
train_size = int(0.8 * len(tokenized_dataset))
train_dataset = tokenized_dataset.select(range(train_size))
eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

# === 6. Define metrics ===
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# === 7. Setup training arguments ===
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    save_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
)

# === 8. Initialize Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# === 9. Train the model ===
trainer.train()

# === 10. Evaluate model performance ===
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# === 11. Create a pipeline for inference using fine-tuned model ===
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=trainer.model,
    tokenizer=tokenizer,
    return_all_scores=False
)

# === 12. Predict function for new text input ===
def predict_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label_map = {'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'}
    label = label_map.get(result['label'], result['label'])
    score = result['score']
    print(f"Input Text: {text}\nPredicted Sentiment: {label} (Confidence: {score:.2f})\n")

# === Example usage ===
if _name_ == "_main_":
    while True:
        user_input = input("Enter a social media text to classify (or 'exit' to quit):\n")
        if user_input.lower() == 'exit':
            break
        predict_sentiment(user_input)
