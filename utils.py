import numpy as np
import copy
import os
import logging


class Metrics:
    """
    Metrics class for calculating accuracy and other scores.
    """
    def __init__(self, label_names):
        """
        Args:
          label_names: list

        Returns:
        """
        self.label_names = label_names

        # Init vars for accuracy
        self.accuracy, self.accuracy_sum = 0, 0

        # Init dict for current metrics (precision, recall, f1)
        self.pres_rec_f1 = {}

        for label_names_i in self.label_names:
            self.pres_rec_f1[label_names_i] = {}

            self.pres_rec_f1[label_names_i]['precision'] = 0
            self.pres_rec_f1[label_names_i]['recall'] = 0
            self.pres_rec_f1[label_names_i]['f1'] = 0

         # Init dict for all metrics sum
        self.pres_rec_f1_sum = copy.deepcopy(self.pres_rec_f1)

    def calculate(self, logit, label):
        """Calculate accuracy, precision, recall, f1 scores.

        Args:
          logit: list
          label: list

        Returns:
            accuracy: int
            pres_rec_f1: dict
        """
        logit = logit.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        # Flatten arrays
        logit_flat = np.argmax(logit, axis=1).flatten()
        label_flat = label.flatten()

        # Calculate accuracy
        self.accuracy = np.mean(logit_flat == label_flat)

        # Calculate precision, recall, f1 for each label name
        for label_names_i in self.label_names:
            precision = self.get_precision(label_flat, logit_flat, label_names_i)
            recall = self.get_recall(label_flat, logit_flat, label_names_i)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

            self.pres_rec_f1[label_names_i] = {
                'precision': round(precision, 2),
                'recall': round(recall, 2),
                'f1': round(f1, 2)
                }

        return self.accuracy, self.pres_rec_f1

    def accumulate(self, accuracy, pres_rec_f1):
        """Accumulate accuracy, precision, recall, f1 scores.

        Args:
          accuracy: int
          pres_rec_f1: dict

        Returns:
            accuracy_sum: int
            pres_rec_f1_sum: dict
        """
        accuracy = accuracy or self.accuracy
        pres_rec_f1 = pres_rec_f1 or self.pres_rec_f1

        # Accumulate accuracy
        self.accuracy_sum += accuracy

        # Accumulate precision, recall, f1 for each label name
        for label_names_i in self.label_names:
            self.pres_rec_f1_sum[label_names_i]['precision'] += pres_rec_f1[label_names_i]['precision']
            self.pres_rec_f1_sum[label_names_i]['recall'] += pres_rec_f1[label_names_i]['recall']
            self.pres_rec_f1_sum[label_names_i]['f1'] += pres_rec_f1[label_names_i]['f1']

        return self.accuracy_sum, self.pres_rec_f1_sum

    def mean_values(self, iter_count):
        """Calculate mean values of accuracy, precision, recall, f1 scores.

        Args:
          iter_count: int

        Returns:
            accuracy_sum: int
            pres_rec_f1_sum: dict
        """
        self.accuracy_sum /= iter_count

        # Accumulate precision, recall, f1 for each label name
        for label_names_i in self.label_names:
            self.pres_rec_f1_sum[label_names_i]['precision'] /= iter_count
            self.pres_rec_f1_sum[label_names_i]['recall'] /= iter_count
            self.pres_rec_f1_sum[label_names_i]['f1'] /= iter_count

        return self.accuracy_sum, self.pres_rec_f1_sum

    def get_true_positives(self, label, logit, label_name):
        """Find number of true positives for label_name.

        Args:
          label: list
          logit: list
          label_name: int | str

        Returns:
            true_positives: int
        """
        true_positives = 0
        for i in range(len(label)):
            if logit[i] == label_name and label[i] == logit[i]:
                true_positives += 1

        return true_positives

    def get_precision(self, label, logit, label_name):
        """Calculate precision for label_name.

        Args:
          label: list
          logit: list
          label_name: int | str

        Returns:
            precision: int
        """
        true_positives = self.get_true_positives(label, logit, label_name)
        all_positives = np.where(logit == label_name, 1, 0).sum()
        precision = true_positives / (all_positives + 1e-7)

        return precision

    def get_recall(self, label, logit, label_name):
        """Calculate recall for label_name.

        Args:
          label: list
          logit: list
          label_name: int | str

        Returns:
            recall: int
        """
        true_positives = self.get_true_positives(label, logit, label_name)
        all_true_values = np.where(label == label_name, 1, 0).sum()
        recall = true_positives / (all_true_values + 1e-7)

        return recall

    def get_f1_score(self, label, logit, label_name):
        """Calculate f1 for label_name.

        Args:
          label: list
          logit: list
          label_name: int | str

        Returns:
            f1: int
        """
        precision = self.get_precision(label, logit, label_name)
        recall = self.get_recall(label, logit, label_name)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

        return f1


def create_logging(log_dir, filemode):
    """Create logging folder and config.
    Args:
      log_dir: str
      filemode: str
    Returns:
      logging
    """
    os.makedirs(log_dir, exist_ok=True)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging



