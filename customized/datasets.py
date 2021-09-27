import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset


class TrainValDataset(Dataset):
    def __init__(self, data, labels, context=0):

        # read in data from file
        # build single numpy array out of utterances
        self.data = np.concatenate(np.load(data, allow_pickle=True), axis=0)
        self.labels = np.concatenate(np.load(labels, allow_pickle=True), axis=0)

        # convert to tensor
        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels)

        # set correct datatypes
        self.data = self.data.type(torch.FloatTensor)
        self.labels = self.labels.type(torch.LongTensor)

        # validate data matches labels
        assert (len(self.data) == len(self.labels))

        # initialize other fields
        self.length = len(self.labels)
        self.context = int(context)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        t_index = index - self.context  # top index
        b_index = index + self.context  # bottom index

        if t_index < 0:  # pad above

            p_size = -t_index
            item = self.data[0:b_index + 1]
            p2d = (0, 0, p_size, 0)  # pad columns by (0, 0) and rows by (p_size, 0)
            item = F.pad(item, p2d, mode='constant', value=0)

        elif b_index > self.length - 1:  # pad below

            p_size = b_index - self.length + 1
            item = self.data[t_index:self.length]
            p2d = (0, 0, 0, p_size)  # pad columns by (0, 0) and rows by (0, p_size)
            item = F.pad(item, p2d, mode='constant', value=0)

        else:  # no padding required
            item = self.data[t_index:b_index + 1]

        label = self.labels[index]
        return item.flatten(), label


class TestDataset(Dataset):
    def __init__(self, data, context=0):

        # read in data from file
        # build single numpy array out of utterances
        self.data = np.concatenate(np.load(data, allow_pickle=True), axis=0)

        # convert to tensor
        self.data = torch.tensor(self.data)

        # set correct datatypes
        self.data = self.data.type(torch.FloatTensor)

        # initialize other fields
        self.length = len(self.data)
        self.context = int(context)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        t_index = index - self.context  # top index
        b_index = index + self.context  # bottom index

        if t_index < 0:  # pad above

            p_size = -t_index
            item = self.data[0:b_index + 1]
            p2d = (0, 0, p_size, 0)  # pad columns by (0, 0) and rows by (p_size, 0)
            item = F.pad(item, p2d, mode='constant', value=0)

        elif b_index > self.length - 1:  # pad below

            p_size = b_index - self.length + 1
            item = self.data[t_index:self.length]
            p2d = (0, 0, 0, p_size)  # pad columns by (0, 0) and rows by (0, p_size)
            item = F.pad(item, p2d, mode='constant', value=0)

        else:  # no padding required
            item = self.data[t_index:b_index + 1]

        return item.flatten()


def convert_output(out):
    # convert 2D output to 1D a single class label (71 nodes into a single number per output)
    out = np.argmax(out, axis=1)  # column with max value in each row is the index of the predicted label

    return out


def acc_func(out, actual):
    """
    out: 2D tensor (torch.FloatTensor), each row has 71 columns (one for each possible label)
    actual: 1D tensor (torch.LongTensor)
    """
    # retrieve labels from device by converting to numpy arrays
    actual = actual.cpu().detach().numpy()

    # convert output to class labels
    pred = convert_output(out)

    # compare predictions against actual
    n_hits = np.sum(pred == actual)

    return n_hits