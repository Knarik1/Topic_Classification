import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
	"""Dataset for onset detection """
	def __init__(self, path_to_folder):
		"""
		Args:
			path_to_folder: str
		"""
		super(CustomDataset, self).__init__()
		
		colab_path = './drive/MyDrive/TopicClassification/'
		
		# Load data into dataFrame
		self.df = pd.read_csv(colab_path + path_to_folder + '/data.txt', sep='\t', names=["label", "text"])
		self.labels	= torch.tensor(self.df['label'])

		# Load token ids
		with open(colab_path + path_to_folder + '/token_ids.json') as json_file:
			token_ids = json.load(json_file)
			self.token_ids = torch.Tensor(token_ids)

		# Load data attention masks
		with open(colab_path + path_to_folder + '/attention_masks.json') as json_file:
			attention_masks = json.load(json_file)
			self.attention_masks = torch.Tensor(attention_masks)

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		"""
		Args:
			index: int

		Returns:
		{
			'token_id': [int],
			'attention_mask': [int],
			'label': [int]
		}: dict
		"""
		label = self.labels[index]
		token_id = self.token_ids[index].int()
		attention_mask = self.attention_masks[index]

		return {'token_id': token_id, 'attention_mask': attention_mask, 'label': label}

	def get_each_label_counts(self):
		""" Count each label frequency count.
		Returns:
			label_counts: pd.Series
		"""
		label_counts = self.df['label'].value_counts().sort_index(axis=0)

		return label_counts

	def get_label_names(self):
		""" Get unique label names.
		Returns:
			label_names: np.array
		"""
		label_names = np.sort(self.df['label'].unique())

		return label_names

