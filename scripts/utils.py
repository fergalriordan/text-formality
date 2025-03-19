import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load data from Hugging Face dataset
def load_data():
    splits = {'train': 'train.csv', 'test': 'test.csv'}
    train_df = pd.read_csv("hf://datasets/osyvokon/pavlick-formality-scores/" + splits["train"])
    test_df = pd.read_csv("hf://datasets/osyvokon/pavlick-formality-scores/" + splits["test"])

    return train_df, test_df

# Predict formality using a Hugging Facemodel
def predict_formality(model, tokenizer, df, return_token_type_ids=True, truncate=True, padding="max_length", batch_size=4):
    id2formality = {0: "formal", 1: "informal"} # from model documentation on Hugging Face

    predicted_labels = [] # 0 for informal, 1 for formal (consistent with dataset labels)
    predicted_logits = [] # store the confidence scores for each prediction
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    for i in tqdm(range(0, len(df), batch_size)):
        texts = df['sentence'][i:i + batch_size].tolist()

        # prepare the input
        encoding = tokenizer(
            texts,
            add_special_tokens=True,
            return_token_type_ids=return_token_type_ids,
            truncation=truncate,
            padding=padding,
            return_tensors="pt",
        )
        
        # Move input tensors to the same device as the model
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # inference
        output = model(**encoding)

        batch_predicted_labels = []
        batch_predicted_logits = []
        for text_scores in output.logits.softmax(dim=1):
            # Move logits to CPU for post-processing
            text_scores = text_scores.cpu()
            score_dict = {id2formality[idx]: score for idx, score in enumerate(text_scores.tolist())}
            predicted_label = 1 if score_dict['formal'] > score_dict['informal'] else 0
            batch_predicted_labels.append(predicted_label)
            batch_predicted_logits.append(score_dict['formal'] if predicted_label == 1 else score_dict['informal'])

        predicted_labels.extend(batch_predicted_labels)
        predicted_logits.extend(batch_predicted_logits)

    return predicted_labels, predicted_logits

# Calculate Confusion Matrix (Binary Classification)
def calculate_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

# Calculate Metrics (Binary Classification)
def calculate_metrics(y_true, y_pred):
    # Calculate metrics for binary classification
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return accuracy, precision, recall, f1

# Display metrics in a visually appealing way
def display_metrics(model_name, cm, accuracy, precision, recall, f1):
    # Create metrics table
    metrics_table = [
        ["Accuracy", f"{accuracy:.4f}"],
        ["Precision", f"{precision:.4f}"],
        ["Recall", f"{recall:.4f}"],
        ["F1 Score", f"{f1:.4f}"]
    ]
    
    print(f"\n{model_name} Performance Metrics:")
    print("="*50)
    print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="fancy_grid"))
    
    # Display confusion matrix as a heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Informal (0)', 'Formal (1)'],
                yticklabels=['Informal (0)', 'Formal (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    plt.tight_layout()
    plt.show()