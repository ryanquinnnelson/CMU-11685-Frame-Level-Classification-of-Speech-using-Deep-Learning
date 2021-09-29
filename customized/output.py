import numpy as np
import pandas as pd


def _convert_output(out):
    # convert 2D output to 1D a single class label (71 nodes into a single number per output)
    out = np.argmax(out, axis=1)  # column with max value in each row is the index of the predicted label

    return out


def format_output(out):
    # convert output to class labels
    converted = _convert_output(out)

    # read in file
    df = pd.DataFrame(converted).reset_index(drop=False)

    # change column names
    df = df.rename(columns={0: "label", 'index': 'id'})

    return df


def evaluate_batch(out, actual):
    """
    out: 2D tensor (torch.FloatTensor), each row has 71 columns (one for each possible label)
    actual: 1D tensor (torch.LongTensor)
    """
    # retrieve labels from device by converting to numpy arrays
    actual = actual.cpu().detach().numpy()

    # convert output to class labels
    pred = _convert_output(out)

    # compare predictions against actual
    n_hits = np.sum(pred == actual)

    return n_hits


def evaluate_epoch(batches_sum, batch_size, dataset_size):
    return batches_sum / dataset_size
