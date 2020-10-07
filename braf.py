import numpy as np
from random import seed
import argparse

import lime
# from lime import submodular_pick
import lime.lime_tabular


from data import load_data, split_data, \
    pima_training_data_transformation, pima_test_data_transformation

from trainer import evaluate_algorithm, biased_random_forest, train_biased_random_forest, test_random_forest, \
    BiasedRandomForestModel
from metrics import display_metrics
from plotting import save_prc_curve, save_roc_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", default="", help="Job directory for output plots")
    parser.add_argument("--data-path", default="diabetes.csv", help="Data directory for PIMA")
    parser.add_argument("--n-folds", type=int, default=10, help="Number of Folds for K-Fold Cross Validation")
    parser.add_argument("--n-trees", type=int, default=100, help="Number of Trees")
    parser.add_argument("--n-neighbors", type=int, default=10, help="Number of neighbors for critical set")
    parser.add_argument("--max-depth", type=int, default=10, help="Depth of search for random forest")
    parser.add_argument("--min-size", type=int, default=1, help="Minimum size for random forest")
    parser.add_argument("--n-features", type=int, default=2, help="Number of features for random forest")
    parser.add_argument("--sample-size", type=float, default=1.0, help="Sample size for random forest")
    parser.add_argument("--p-critical", type=float, default=0.5, help="Percentage of forest size using critical set")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # Set seeds for reproducibility
    np.random.seed(seed=args.seed)
    seed(args.seed)

    # Prep data for model training
    data = load_data(args.data_path)
    raw_data_train, raw_data_test = split_data(data)
    data_train, medians = pima_training_data_transformation(raw_data_train)

    # Evaluate algorithm on training data using K-Fold cross validation
    _ = evaluate_algorithm(data_train, biased_random_forest, args.n_folds,
                           args.n_neighbors, args.p_critical, args.max_depth, args.min_size,
                           args.sample_size, args.n_trees, args.n_features
                           )

    # Train tree model on full training dataset
    trees = train_biased_random_forest(data_train, args.n_neighbors, args.max_depth, args.min_size,
                                       args.sample_size, args.n_trees,
                                       args.n_features, args.p_critical)

    # Evaluate model on test data
    # Prepare test data
    data_test = pima_test_data_transformation(raw_data_test, medians).to_numpy()

    test_set = list()
    for row in data_test:
        row_copy = list(row)
        test_set.append(row_copy)
        row_copy[-1] = None

    # Run inference on test set
    test_predictions, test_probs = test_random_forest(trees, test_set)
    test_actual = data_test[:, -1]

    # Evaluate test data performance
    print('Test Data Performance')
    fp_rates, tp_rates, recalls, precisions = display_metrics(test_actual, test_predictions, test_probs)

    # Plot final
    outname = "Test Data"
    save_prc_curve(recalls, precisions, name=outname)
    save_roc_curve(fp_rates, tp_rates, name=outname)

    # LIME
    df_features = data_train.iloc[:, :-1]
    feature_cols = df_features.columns
    data_features = df_features.values
    data_labels = data_train.iloc[:, -1].values

    explainer = lime.lime_tabular.LimeTabularExplainer(data_features,
                                                       mode='classification', training_labels=data_labels,
                                                       feature_names=feature_cols)

    model = BiasedRandomForestModel(trees)

    idx = 0
    exp = explainer.explain_instance(data_features[idx], model.get_probs, num_features=7)
    exp.save_to_file('lime_rf_example0.html')
    # sp_obj = submodular_pick.SubmodularPick(explainer, data_features, model.get_probs,
    #                                         sample_size=20, num_features=7, num_exps_desired=5)

    # [exp.as_pyplot_figure() for exp in sp_obj.sp_explanations]


if __name__ == "__main__":
    main()
