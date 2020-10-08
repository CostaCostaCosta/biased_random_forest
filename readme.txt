Numpy implementation of Biased Random Forest:
https://ieeexplore.ieee.org/document/8541100

Runs default arguments with:
python -m braf

ARGS:
("--job-dir", default=None, help="Job directory for output plots")
("--data-path", default="diabetes.csv", help="Data directory for PIMA")
("--n-folds", type=int, default=10, help="Number of Folds for K-Fold Cross Validation")
("--n-trees", type=int, default=100, help="Number of Trees")
("--n-neighbors", type=int, default=10, help="Number of neighbors for critical set")
("--max-depth", type=int, default=10, help="Depth of search for random forest")
("--min-size", type=int, default=1, help="Minimum size for random forest")
("--n-features", type=int, default=2, help="Number of features for random forest")
("--sample-size", default=1.0, help="Sample size for random forest")
("--p_critical", default=0.5, help="Percentage of forest size using critical set")
("--seed", type=int, default=0, help="Random seed")