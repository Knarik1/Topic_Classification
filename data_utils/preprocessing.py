import os
import json
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer


def tokenize(path):
    """
    Args:
        path: str (Path to data file)
    """
    print('Loading data...')
    df = pd.read_csv(path, sep="\t", names=["label", "text"])

    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Tokenize all of the sentences and map the tokens to their word IDs.
    token_ids = []
    attention_masks = []

    for pgh in tqdm(df['text']):
        # For every paragraph...
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            pgh,
                            add_special_tokens=True,
                            truncation=True,
                            padding='max_length',
                            max_length=128,
                            return_attention_mask=True,
                            return_tensors='np',
                       )

        # Add the encoded sentence to the list.
        token_ids.append(encoded_dict['input_ids'].squeeze(0).tolist())

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'].squeeze(0).tolist())

    # Save to JSONs
    dirname = os.path.dirname(path)
    token_ids_path = os.path.join(dirname, "token_ids.json")
    attention_masks_path = os.path.join(dirname, "attention_masks.json")

    with open(token_ids_path, 'w') as f:
        json.dump(token_ids, f)
        print('Saved to', token_ids_path)

    with open(attention_masks_path, 'w') as f:
        json.dump(attention_masks, f)
        print('Saved to', attention_masks_path)


def split_data(path):
    """ Stratified split data into train/valid/test and save in folders
    Args:
        path: str (Path to directory data file)
    """
    os.makedirs('./data/train', exist_ok=True)
    os.makedirs('./data/valid', exist_ok=True)
    os.makedirs('./data/test', exist_ok=True)

    df = pd.read_csv(path, delimiter=" ", names=["label", "text"])


    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'],
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=42,
                                                        stratify=df['label'])

    # Stratified split
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=42,
                                                        stratify=y_train)

    train_df = pd.DataFrame(columns=["label", "text"])
    train_df['label'] = y_train
    train_df['text'] = X_train

    valid_df = pd.DataFrame(columns=["label", "text"])
    valid_df['label'] = y_valid
    valid_df['text'] = X_valid

    test_df = pd.DataFrame(columns=["label", "text"])
    test_df['label'] = y_test
    test_df['text'] = X_test

    # Save data to csv files
    train_df.to_csv('./data/train/data.txt', header=None, index=None, sep='\t', mode='a', doublequote=False, escapechar=' ')
    valid_df.to_csv('./data/valid/data.txt', header=None, index=None, sep='\t', mode='a', doublequote=False, escapechar=' ')
    test_df.to_csv('./data/test/data.txt', header=None, index=None, sep='\t', mode='a', doublequote=False, escapechar=' ')


if __name__ == "__main__":

    # Split data into train/valid/test folders
    split_data('./data/train_cleaned.txt')

    # Tokenize each split data
    for folder in ['train', 'valid', 'test']:
        print("Folder", folder)
        tokenize('./data/' + folder + '/data.txt')
