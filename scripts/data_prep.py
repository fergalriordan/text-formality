import pandas as pd

def load_data(threshold=1.0):
    splits = {'train': 'train.csv', 'test': 'test.csv'}
    train_df = pd.read_csv("hf://datasets/osyvokon/pavlick-formality-scores/" + splits["train"])
    test_df = pd.read_csv("hf://datasets/osyvokon/pavlick-formality-scores/" + splits["test"])
    binary_df = test_df[test_df['avg_score'].abs() > threshold]

    # create a binary column for formality
    binary_df['formal'] = binary_df['avg_score'].apply(lambda x: 1 if x > 0 else 0)

    return train_df, test_df, binary_df
