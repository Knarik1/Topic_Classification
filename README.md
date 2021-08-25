# Topic_Classification

The goal is to classify documents for the given dataset.

## Description of the work
  * data exploration and removement of duplicate rows and short text data, from which it is harder for even a human to classify the text
  * data split(stratified) into train/valid/test 
  * used HuggingFace transformers Bert tokenizer with maximum text length of 128 tokens, as it's assumed that it's enough to understand the topic from that
  size lenght and token ids and attention masks were saved for later use
  * for pretrained model wass taken BertForSequenceClassification and continued to train for 3 epochs on the dataset
  * as the dataset was imbalanced used CrossEntropyLoass with already calculated label weights
  * for calculating metrics the Metric class was created, which ease calculations of scores (precision, recall, f1) for each class
  * you can see the results and error analysis for test data in run.py file prints

## Running Environment 
  * Google Colab with GPU device
  
## Results  
  ![Alt text](https://github.com/Knarik1/Topic_Classification/blob/main/results/Screenshot%20from%202021-08-09%2010-24-44.png)
