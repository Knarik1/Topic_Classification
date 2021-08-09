import random
import argparse
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from aim import Session
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup

from data_utils.dataset import CustomDataset
from utils import Metrics, create_logging


def train_f(model, train_dataloader, criterion, optimizer, scheduler, sess, device):
	"""Train model and calculate output scores.

    Args:
      model: transformers.BertForSequenceClassification
      train_dataloader: torch.DataLoader
      criterion: torch.nn._Loss
      optimizer: torch.Optimizer
      scheduler: torch.optim.lr_scheduler
      device: 'cuda' | 'cpu'

    Returns:
    	train_loss: int
    	accuracy: int
    	other_metrics: dict
    """
	# Init loss value
	train_loss = 0

	# Get label unique values and init metrics class
	label_names = train_dataloader.dataset.get_label_names()
	metrics = Metrics(label_names)

	# Train mode
	model.train()

	for i, data_dict in enumerate(tqdm(train_dataloader)):
		# Get data
		token_id = data_dict['token_id'].to(device)
		attention_mask = data_dict['attention_mask'].to(device)
		label = data_dict['label'].to(device)

		# Forward propagation
		output = model(token_id,
					   token_type_ids=None,
					   attention_mask=attention_mask,
                       labels=label,
                       return_dict=True)

		# Get logits
		logit = output.logits

		# Calculate loss
		loss = criterion(logit, label)

		# Put to zero old gradients
		optimizer.zero_grad()

		# Backward propagation and gradient clipping
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

		# Update
		optimizer.step()
		scheduler.step()

		# Metrics calculate accuracy, precision, recall, f1 scores
		accuracy, other_metrics = metrics.calculate(logit, label)

		# Accumulate
		train_loss += loss.item()
		metrics.accumulate(accuracy, other_metrics)

		if i % 200 == 0:
			print('\nIteration {}, Train Loss: {:.4f}'.format(i, loss.item()))

	# Mean values
	num_iter = len(train_dataloader)
	train_loss /= num_iter
	accuracy, other_metrics = metrics.mean_values(num_iter)

	# Aim track
	sess.track(train_loss, name='loss', epoch=i, subset='train')
	sess.track(accuracy, name='accuracy', epoch=i, subset='train')

	return train_loss, accuracy, other_metrics


def valid_f(model, valid_dataloader, criterion, sess, device):
	"""Validate model and calculate output scores.

    Args:
      model: transformers.BertForSequenceClassification
      valid_dataloader: torch.DataLoader
      criterion: torch.nn._Loss
      device: 'cuda' | 'cpu'

    Returns:
    	valid_loss: int
    	accuracy: int
    	other_metrics: dict
    """
	# Init loss value
	valid_loss = 0

	# Get label unique values and init metrics class
	label_names = valid_dataloader.dataset.get_label_names()
	metrics = Metrics(label_names)

	# Eval mode
	model.eval()

	with torch.no_grad():
		for i, data_dict in enumerate(tqdm(valid_dataloader)):
		# Get data
			token_id = data_dict['token_id'].to(device)
			attention_mask = data_dict['attention_mask'].to(device)
			label = data_dict['label'].to(device)

			# Forward propagation
			output = model(token_id,
						   token_type_ids=None,
						   attention_mask=attention_mask,
						   labels=label,
						   return_dict=True)

			# Get logits
			logit = output.logits

			# Calculate loss
			loss = criterion(logit, label)

			# Metrics calculate accuracy, precision, recall, f1 scores
			accuracy, other_metrics = metrics.calculate(logit, label)

			# Accumulate
			valid_loss += loss.item()
			metrics.accumulate(accuracy, other_metrics)

			if i % 100 == 0:
				print('\nIteration {}, Valid Loss: {:.4f}'.format(i, loss.item()))

		# Mean values
		num_iter = len(valid_dataloader)
		valid_loss /= num_iter
		accuracy, other_metrics = metrics.mean_values(num_iter)

		# Aim track
		sess.track(valid_loss, name='loss', epoch=i, subset='valid')
		sess.track(accuracy, name='accuracy', epoch=i, subset='valid')

	return valid_loss, accuracy, other_metrics


def train(args):
	"""Train a topic classification model.

    Args:
      batch_size: int
      epochs: int
      learning_rate: float
      resume: bool
      model_path: str
      cuda: bool
    """
	# Arguments and parameters
	batch_size = args.batch_size
	epochs = args.epochs
	learning_rate = args.learning_rate
	model_path = args.model_path
	resume = args.resume
	start_epoch = 0
	device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else torch.device('cpu'))

	# Set the seed value to make this reproducible.
	seed_val = 7
	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)

	# Logging
	logs_dir = './logging'
	create_logging(logs_dir, filemode='w')
	logging.info(args)
	logging.info('Using device {}'.format(device))

	# Aim - track hyper parameters
	sess = Session(experiment='BERT_finetune')
	sess.set_params({
		'num_epochs': epochs,
		'num_classes': 4,
		'batch_size': batch_size,
		'learning_rate': learning_rate,
		'model_name': model_path
		},
		name='hparams')

	#================================================#
	#                  Data config                   #
	#================================================#

	# Create custom datasets and dataLoaders
	train_dataset = CustomDataset(path_to_folder='./data/train')
	valid_dataset = CustomDataset(path_to_folder='./data/valid')

	print("Train data size", len(train_dataset))
	print("Valid data size", len(valid_dataset))

	train_dataloader = DataLoader(
            train_dataset,
            batch_size = batch_size,
			shuffle=True
        )
	valid_dataloader = DataLoader(
            valid_dataset,
            batch_size = batch_size,
			shuffle=False
        )

	#================================================#
	#                  Model config                  #
	#================================================#
	# Get unique label names
	label_names = train_dataloader.dataset.get_label_names()

	# Create the model and set to device
	model = BertForSequenceClassification.from_pretrained(
		"bert-base-uncased",
		num_labels=len(label_names),
		output_attentions=False,
		output_hidden_states = False
	)
	model = model.to(device)

	#================================================#
	#                  Train config                  #
	#================================================#

	# An improved version of Adam that converges much faster
	optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

	# Total number of training steps is [number of batches] x [number of epochs].
	total_steps = len(train_dataloader) * epochs

	# Create the learning rate scheduler.
	scheduler = get_linear_schedule_with_warmup(optimizer,
												num_warmup_steps = 0,
												num_training_steps = total_steps)

	# Get each label/class count
	label_counts = train_dataset.get_each_label_counts()
	print("Label counts in train dataset.")
	print(label_counts)

	# Normalize counts and put in Tensor
	label_weights = [1 - (c / np.sum(label_counts)) for c in label_counts]
	label_weights = torch.FloatTensor(label_weights).to(device)

	# Loss function with label weights
	criterion = nn.CrossEntropyLoss(weight=label_weights)

	# Loading from the saved model
	if resume:
		print("Loading checkpoint from {} ...".format(model_path))
		logging.info('Loading checkpoint {}'.format(model_path))

		checkpoint = torch.load(model_path, map_location=device)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		loss = checkpoint['loss']
		start_epoch = checkpoint['epoch']
		epochs += start_epoch

	#================================================#
	#             Training and validating            #
	#================================================#

	loss_train = []
	loss_valid = []

	for epoch in range(start_epoch, epochs):
		print("============================ Start of epoch ", epoch, " ===================================")
		train_loss, train_acc, other_metrics = train_f(model, train_dataloader, criterion, optimizer, scheduler, sess, device)
		print('Train Mean Loss: {:.4f}, \n Mean Acc: {:.2f}'.format(epoch, train_loss, train_acc))
		print(other_metrics)

		print("===========================================================================================")

		valid_loss, valid_acc, valid_other_metrics = valid_f(model, valid_dataloader, criterion, sess, device)
		print('Valid Mean Loss: {:.4f}, \n Mean Acc: {:.2f}'.format(epoch, valid_loss, valid_acc))
		print(valid_other_metrics)

		logging.info('--------------------------------------------------------')
		logging.info('Epoch: {}'.format(epoch))
		logging.info('Train Mean Loss: {:.4f}, \n Mean Acc: {:.2f}'.format(epoch, train_loss, train_acc))
		logging.info(other_metrics)
		logging.info('Valid Mean Loss: {:.4f}, \n Mean Acc: {:.2f}'.format(epoch, valid_loss, valid_acc))
		logging.info(valid_other_metrics)


	print("Training is done.")

	#================================================#
	#                Saving results                  #
	#================================================#

	# Save the model with all parameters
	torch.save({
				'epoch': epochs,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': loss_train},
		model_path)

	print("Model saved at", model_path)


def test(args):
	"""Test a topic classification model.

    Args:
      model_path: str
      batch_size: int
      cuda: bool
    """
	# Arguments and parameters
	model_path = args.model_path
	batch_size = args.batch_size
	device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else torch.device('cpu'))

	# Set the seed value to make this reproducible.
	seed_val = 7
	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)

	#================================================#
	#                  Data config                   #
	#================================================#

	# Create custom datasets and dataLoaders
	test_dataset = CustomDataset(path_to_folder='./data/test')

	print("Test data size", len(test_dataset))

	test_dataloader = DataLoader(
            test_dataset,
            batch_size = batch_size,
			shuffle=False
        )

	#================================================#
	#                  Model config                  #
	#================================================#

	# Create the model and set to device
	model = BertForSequenceClassification.from_pretrained(
		"bert-base-uncased",
		num_labels = 4,
		output_attentions = False,
		output_hidden_states = False
	)
	model = model.to(device)

	print("Loading checkpoint from {} ...".format(model_path))
	checkpoint = torch.load(model_path, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])

	#================================================#
	#                  Model testing                 #
	#================================================#

	# Eval mode
	model.eval()

	# Get label unique values and init metrics class
	label_names = test_dataloader.dataset.get_label_names()
	metrics = Metrics(label_names)

	# For error analysis
	logit_arr = []

	with torch.no_grad():
		for i, data_dict in enumerate(tqdm(test_dataloader)):
		# Get data
			token_id = data_dict['token_id'].to(device)
			attention_mask = data_dict['attention_mask'].to(device)
			label = data_dict['label'].to(device)

			# Forward propagation
			output = model(token_id,
						   token_type_ids=None,
						   attention_mask=attention_mask,
						   labels=label,
						   return_dict=True)

			logit = output.logits

			# Metrics calculate accuracy, precision, recall, f1 scores
			accuracy, other_metrics = metrics.calculate(logit, label)

			# Accumulate
			metrics.accumulate(accuracy, other_metrics)

			# Append logits for error analysis
			logit = logit.detach().cpu().numpy()
			logit_flat = np.argmax(logit, axis=1).flatten()
			logit_arr.append(logit_flat)

		# Mean values
		num_iter = len(test_dataloader)
		accuracy, other_metrics = metrics.mean_values(num_iter)

		print("\n==================================== Results ===========================================")
		print("Accuracy is ", accuracy)
		print("Other metrics")
		df = pd.DataFrame(columns=["Label", "Precision", "Recall", "F1"])

		for i, label_names_i in enumerate(test_dataloader.dataset.get_label_names()):
			# Adding values to dataFrame row
			df.loc[-1] = [label_names_i, other_metrics[label_names_i]['precision'], other_metrics[label_names_i]['recall'], other_metrics[label_names_i]['f1']]
			df.index += 1

		print(df.to_string(index=False))

		print("\n================================= Error Analysis ======================================")
		logit_arr = np.concatenate(logit_arr, axis=None)

		# Add pediction column to test data dataFrame
		test_dataset.df.insert(1, "prediction", logit_arr)

		# Get rows where label is not equal to prediction
		df_not_equal_data = test_dataset.df.loc[~(test_dataset.df['label'] == test_dataset.df['prediction'])]
		print(df_not_equal_data.head())


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Parser')
	subparsers = parser.add_subparsers(dest='mode')

	parser_train = subparsers.add_parser('train')
	parser_train.add_argument('--batch_size', type=int, required=True)
	parser_train.add_argument('--epochs', type=int, required=True)
	parser_train.add_argument('--learning_rate', type=float, required=True)
	parser_train.add_argument('--model_path', type=str, required=True)
	parser_train.add_argument('--resume', action='store_true', default=False)
	parser_train.add_argument('--cuda', action='store_true', default=False)

	parser_test = subparsers.add_parser('test')
	parser_test.add_argument('--model_path', type=str, required=True)
	parser_test.add_argument('--batch_size', type=int, required=True)
	parser_test.add_argument('--cuda', action='store_true', default=False)

	args = parser.parse_args()

	if args.mode == 'train':
		train(args)
	elif args.mode == 'test':
		test(args)
	else:
		raise Exception('Error argument!')
