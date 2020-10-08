import numpy as np
from random import randrange
import metrics
import pandas as pd
import ipdb

from plotting import save_prc_curve, save_roc_curve


# find critical data using knn
def find_critical_data(dataset, n_neighbors):
	# split training data into minority/majority classes
	train_set_df = pd.DataFrame(dataset)
	data_train_majority = train_set_df.loc[train_set_df[train_set_df.columns[-1]] == 0].values
	data_train_minority = train_set_df.loc[train_set_df[train_set_df.columns[-1]] == 1].values

	# find critical data with knn
	critical_majority_ind = np.array([], dtype='int')
	for min_point in data_train_minority:
		critical_majority_ind = np.append(critical_majority_ind, kneighbors(min_point, data_train_majority, n_neighbors))

	# remove duplicates from knn output
	critical_majority_ind_unique = np.unique(critical_majority_ind)

	# select relevant data from majority dataset
	critical_majority = data_train_majority[critical_majority_ind_unique]

	# combine minority and critical majority to create critical dataset
	critical_set = np.append(data_train_minority, critical_majority, axis=0)
	return critical_set


# Split a dataset into k folds
# https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = dataset.values.tolist()
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, n_neighbors, p_critical, *args):
	recall_list = list()
	precision_list = list()
	fp_rates_list = list()
	tp_rates_list = list()

	#  K fold cross validation
	folds = cross_validation_split(dataset, n_folds)

	for i, fold in enumerate(folds, start=1):
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()

		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None

		# generate critical set for biased random forest
		critical_set = find_critical_data(train_set, n_neighbors)

		# train braf algorithm
		prediction, prob, trees = algorithm(train_set, critical_set, test_set, p_split=p_critical, *args)
		actual = [row[-1] for row in fold]

		# print key metrics for each fold
		print('Fold: %d' % i)
		fp_rates, tp_rates, recalls, precisions = metrics.display_metrics(actual, prediction, prob)

		# Update lists for plotting
		recall_list.append(recalls)
		precision_list.append(precisions)
		fp_rates_list.append(fp_rates)
		tp_rates_list.append(tp_rates)

	# Plot prc and roc curves for training
	for i in range(len(recall_list)):
		outname = 'Fold' + str(i+1)
		save_prc_curve(recall_list[i], precision_list[i], name=outname)

	for i in range(len(fp_rates_list)):
		outname = 'Fold' + str(i+1)
		save_roc_curve(fp_rates_list[i], tp_rates_list[i], name=outname)

	# Can use this to export data for detailed plots of ROC/PRC curves
	# pd.DataFrame(recall_list).to_csv("recalls.csv")
	# pd.DataFrame(precision_list).to_csv("precisions.csv")
	# pd.DataFrame(fp_rates_list).to_csv("fprates.csv")
	# pd.DataFrame(tp_rates_list).to_csv("tprates.csv")

	return trees


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini


# Select the best split point for a dataset
def get_split(dataset, n_features):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	features = list()
	while len(features) < n_features:
		index = randrange(len(dataset[0])-1)
		if index not in features:
			features.append(index)
	for index in features:
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
	left, right = node['groups']
	del (node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left, n_features)
		split(node['left'], max_depth, min_size, n_features, depth + 1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, n_features)
		split(node['right'], max_depth, min_size, n_features, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
	root = get_split(train, n_features)
	split(root, max_depth, min_size, n_features, 1)
	return root


# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']


# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	prediction = max(set(predictions), key=predictions.count)
	# fraction of trees voting for positive class
	prediction_prob = sum(predictions) / len(predictions)
	return prediction, prediction_prob


# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
	trees = list()
	predictions = list()
	probs = list()
	for i in range(n_trees):
		sample = subsample(train, sample_size)
		tree = build_tree(sample, max_depth, min_size, n_features)
		trees.append(tree)

	# generate predictions and probabilities using bag predict
	for row in test:
		prediction, prob = bagging_predict(trees, row)
		predictions.append(prediction)
		probs.append(prob)

	return predictions, probs, trees


# Biased Random Forest Algorithm
def biased_random_forest(train_full, train_critical, test, max_depth, min_size, sample_size, n_trees, n_features, p_split):
	trees = list()
	predictions = list()
	probs = list()

	# calculate split of trees for the full/critical datasets
	n_trees_full = int(n_trees * (1.0 - p_split))
	n_trees_critical = int(n_trees * p_split)

	# Build forest on full train set
	for i in range(n_trees_full):
		sample = subsample(train_full, sample_size)
		tree = build_tree(sample, max_depth, min_size, n_features)
		trees.append(tree)

	# Build forest on critical train set
	for i in range(n_trees_critical):
		sample = subsample(train_critical, sample_size)
		tree = build_tree(sample, max_depth, min_size, n_features)
		trees.append(tree)

	# generate predictions and probabilities using bag predict
	for row in test:
		prediction, prob = bagging_predict(trees, row)
		predictions.append(prediction)
		probs.append(prob)

	return predictions, probs, trees


# Biased Random Forest Training
def train_biased_random_forest(train_set, n_neighbors, max_depth, min_size, sample_size, n_trees, n_features, p_split):
	trees = list()

	# generate critical set for biased random forest
	train_critical = find_critical_data(train_set, n_neighbors)

	# calculate split of trees for the full/critical datasets
	n_trees_full = int(n_trees * (1 - p_split))
	n_trees_critical = int(n_trees * p_split)

	# Build forest on full train set
	# Get values of training input
	train_full = train_set.values
	for i in range(n_trees_full):
		sample = subsample(train_full, sample_size)
		tree = build_tree(sample, max_depth, min_size, n_features)
		trees.append(tree)

	# Build forest on critical train set
	for i in range(n_trees_critical):
		sample = subsample(train_critical, sample_size)
		tree = build_tree(sample, max_depth, min_size, n_features)
		trees.append(tree)

	return trees


# Random Forest Inference
def test_random_forest(trees, test):
	predictions = list()
	probs = list()

	# generate predictions and probabilities using bag predict
	for row in test:
		prediction, prob = bagging_predict(trees, row)
		predictions.append(prediction)
		probs.append(prob)

	return predictions, probs


# Calculate simple euclidean distance
def euclidian_distance(a, b):
	return np.sqrt(np.sum((a-b)**2, axis=0))


# Numpy implementation of KNN comparing single minor point to dataset, returns indices of the KNN
def kneighbors(min_point, maj_dataset, n_neighbors):

	point_dist = np.asarray([euclidian_distance(min_point, majority_point) for majority_point in maj_dataset])
	idx = np.argpartition(point_dist, n_neighbors)
	neigh_ind = idx[:n_neighbors]

	return neigh_ind


class BiasedRandomForestModel:
	def __init__(self, trees):
		self.trees = trees

	# Probability determination function for LIME
	def get_probs(self, test):
		probs = list()

		# if test is 1d array
		if test.ndim == 1:
			_, prob = bagging_predict(self.trees, test)
			probs.append(prob)

		# if array is not 1d
		else:
		# generate predictions and probabilities using bag predict
			for row in test:
				_, prob = bagging_predict(self.trees, row)
				probs.append(prob)

		probs = np.asarray(probs)
		# Probs must be formatted as [p(class 0), p(class 1)]
		probs_array = np.array(list(zip(1.0 - probs, probs)))
		return probs_array
