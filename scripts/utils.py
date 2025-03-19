import pandas as pd
from tqdm import tqdm

# Load data from Hugging Face dataset
def load_data(threshold=1.0):
    splits = {'train': 'train.csv', 'test': 'test.csv'}
    train_df = pd.read_csv("hf://datasets/osyvokon/pavlick-formality-scores/" + splits["train"])
    test_df = pd.read_csv("hf://datasets/osyvokon/pavlick-formality-scores/" + splits["test"])
    binary_df = test_df[test_df['avg_score'].abs() > threshold]

    # create a binary column for formality
    binary_df['formal'] = binary_df['avg_score'].apply(lambda x: 1 if x > 0 else 0)

    return train_df, test_df, binary_df

# Predict formality using a Hugging Facemodel
def predict_formality(model, tokenizer, df, batch_size=4):
    id2formality = {0: "formal", 1: "informal"} # from model documentation on Hugging Face

    predicted_labels = [] # 0 for informal, 1 for formal (consistent with dataset labels)

    for i in tqdm(range(0, len(df), batch_size)):
        texts = df['sentence'][i:i + batch_size].tolist()

        # prepare the input
        encoding = tokenizer(
            texts,
            add_special_tokens=True,
            return_token_type_ids=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # inference
        output = model(**encoding)

        batch_predicted_labels = []
        for text_scores in output.logits.softmax(dim=1):
            score_dict = {id2formality[idx]: score for idx, score in enumerate(text_scores.tolist())}
            batch_predicted_labels.append(1 if score_dict['formal'] > score_dict['informal'] else 0)

        predicted_labels.extend(batch_predicted_labels)

    return predicted_labels