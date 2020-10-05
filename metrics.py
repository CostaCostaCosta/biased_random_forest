import numpy as np


# Calculate target metrics
def display_metrics(actual, predicted, predicted_probs):

    # determine relevant counts on full dataset
    positive, negative, true_positive, false_positive, false_negative = find_counts(actual, predicted)

    # Calculate precision and recall
    precision_tot = true_positive / float(true_positive + false_positive)
    recall_tot = true_positive / float(true_positive + false_negative)

    # Calculate target rates for each unique value in the output triple
    fp_rates, tp_rates, recalls, precisions = calc_rates(actual, predicted_probs)

    # Calculate AUPRC and AUROC using trapezoidal rule, may overestimate sk-learn calculation
    auprc = -1 * np.trapz(precisions, recalls)
    auroc = np.trapz(tp_rates, fp_rates)

    print('Precision: %.3f' % precision_tot)
    print('Recall: %.3f' % recall_tot)
    print('AUPRC: %.3f' % auprc)
    print('AUROC: %.3f' % auroc)

    return fp_rates, tp_rates, recalls, precisions


# Using previously constructed sorted array of [actual, predicted, probs],
# use thresholding at each unique value to prepare for plotting
# https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/metrics/_ranking.py#L583
def calc_rates(actual, probs):
    actual = np.asarray(actual)
    probs = np.asarray(probs)
    # sort scores
    desc_score_indices = np.argsort(probs, kind="mergesort")[::-1]
    probs = probs[desc_score_indices]
    actual = actual[desc_score_indices]

    # find index of unique probability values
    unique_probs_idxs = np.where(np.diff(probs))[0]
    threshold_idxs = np.r_[unique_probs_idxs, actual.size - 1]

    true_positives = np.cumsum(actual)[threshold_idxs]
    false_positives = np.cumsum(1-actual)[threshold_idxs]

    precisions = true_positives / (true_positives + false_positives)
    precisions[np.isnan(precisions)] = 0
    recalls = true_positives / true_positives[-1]

    # reverse outputs so recall is decreasing
    last_ind = true_positives.searchsorted(true_positives[-1])
    s1 = slice(last_ind, None, -1)
    recalls = np.r_[recalls[s1], 0]  # Add final value 0
    precisions = np.r_[precisions[s1], 1]  # Add final value 1

    true_positives_rates = true_positives / float(true_positives[-1])
    false_positives_rates = false_positives / float(false_positives[-1])

    return false_positives_rates, true_positives_rates, recalls, precisions


# calculates positive, negative, true_positive, false_positive, false_negative counts of the input
def find_counts(actual, predicted):
    positive = 0
    negative = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0

    for i in range(len(actual)):
        if actual[i] == 1:
            positive += 1
        if actual[i] == 0:
            negative += 1
        if actual[i] == 1 and predicted[i] == 1:
            true_positive += 1
        if actual[i] == 1 and predicted[i] == 0:
            false_negative += 1
        if actual[i] == 0 and predicted[i] == 1:
            false_positive += 1

    return positive, negative, true_positive, false_positive, false_negative


# Returns unique rows of a 2d-numpy array
# https://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

