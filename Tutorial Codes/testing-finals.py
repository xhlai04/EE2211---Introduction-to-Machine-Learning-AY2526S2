"""
╔══════════════════════════════════════════════════════════════╗
║              EXAM PLUG-AND-PLAY TOOLKIT                      ║
║  Topics: Correlation | Polynomial Regression | Gradient      ║
║          Descent | Decision Trees | Random Forest |          ║
║          Model Evaluation | K-Means Clustering               ║
╚══════════════════════════════════════════════════════════════╝

HOW TO USE:
  1. Scroll to the section you need (marked with ===).
  2. Replace the sample data / parameters at the top of each block.
  3. Run the block — results are printed automatically.
"""

import numpy as np
from numpy.linalg import inv

# ══════════════════════════════════════════════════════════════
# 1.  PEARSON CORRELATION
# ══════════════════════════════════════════════════════════════
def pearson_correlation(x, y):
    """
    Returns Pearson r between two lists / arrays.
    Usage:
        r = pearson_correlation(feature1, target_y)
    """
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    diff_x = x - x.mean()
    diff_y = y - y.mean()
    return np.dot(diff_x, diff_y) / (np.sqrt(np.sum(diff_x**2) * np.sum(diff_y**2)))


def pearson_all_features(features_dict, target_y):
    """
    Prints Pearson r for every feature vs target.
    Usage:
        features = {"feature1": [0.35, 2.18, ...], "feature2": [...]}
        pearson_all_features(features, target_y)
    """
    target_y = np.array(target_y, dtype=float)
    print("=== Pearson Correlation ===")
    for name, vals in features_dict.items():
        r = pearson_correlation(vals, target_y)
        print(f"  {name:20s}  r = {r:.6f}")


# ─── EXAMPLE (replace with your data) ─────────────────────────
if __name__ == "__main__":
    feature1 = [0.3510, 2.1812, 0.2415, -0.1096, 0.1544]
    feature2 = [1.1796, 2.1068, 1.7753,  1.2747, 2.0851]
    feature3 = [-0.9852, 1.3766, -1.3244, -0.6136, -0.8320]
    target_y  = [0.2758, 1.4392, -0.4611,  0.6154,  1.0006]

    pearson_all_features(
        {"feature1": feature1, "feature2": feature2, "feature3": feature3},
        target_y
    )


# ══════════════════════════════════════════════════════════════
# 2.  POLYNOMIAL REGRESSION  (with & without L2 regularisation)
# ══════════════════════════════════════════════════════════════
def poly_regression(X_tr, y_tr, X_ts, y_ts,
                    max_degree=6, lam=0.0, verbose=True):
    """
    Fits polynomial regression for degrees 1..max_degree using closed-form
    solution (no sklearn needed). Returns MSE lists.

    Parameters
    ----------
    X_tr, X_ts : 1-D or column arrays of input values
    y_tr, y_ts : 1-D or column arrays of targets
    max_degree  : highest degree to try
    lam         : ridge regularisation lambda (0 = no regularisation)
    verbose     : print MSE per degree

    Returns
    -------
    MSE_train_list, MSE_test_list  (lists of floats, one per degree)

    Usage
    -----
    tr_mse, ts_mse = poly_regression(X_tr, y_tr, X_ts, y_ts,
                                     max_degree=6, lam=1.0)
    """
    from sklearn.preprocessing import PolynomialFeatures

    X_tr = np.array(X_tr, dtype=float).reshape(-1, 1)
    X_ts = np.array(X_ts, dtype=float).reshape(-1, 1)
    y_tr = np.array(y_tr, dtype=float).reshape(-1, 1)
    y_ts = np.array(y_ts, dtype=float).reshape(-1, 1)

    MSE_train_list, MSE_test_list = [], []

    label = f"lambda={lam}" if lam > 0 else "no regularisation"
    if verbose:
        print(f"\n=== Polynomial Regression ({label}) ===")

    for deg in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree=deg)
        P_tr = poly.fit_transform(X_tr)
        P_ts = poly.transform(X_ts)
        m, d = P_tr.shape

        if m > d:   # over-determined: normal equations
            A = P_tr.T @ P_tr + lam * np.eye(d)
            w = inv(A) @ P_tr.T @ y_tr
        else:       # under-determined: minimum-norm solution
            A = P_tr @ P_tr.T + lam * np.eye(m)
            w = P_tr.T @ inv(A) @ y_tr

        mse_tr = np.mean((P_tr @ w - y_tr) ** 2)
        mse_ts = np.mean((P_ts @ w - y_ts) ** 2)
        MSE_train_list.append(float(mse_tr))
        MSE_test_list.append(float(mse_ts))

        if verbose:
            print(f"  Degree {deg}:  train MSE = {mse_tr:.6f}  |  test MSE = {mse_ts:.6f}")

    return MSE_train_list, MSE_test_list


# ─── EXAMPLE ──────────────────────────────────────────────────
if __name__ == "__main__":
    X_tr = [-10, -8, -3, -1,  2,  7]
    y_tr = [4.18, 2.42, 0.22, 0.12, 0.25, 3.09]
    X_ts = [-9, -7, -5, -4, -2, 1, 4, 5, 6, 9]
    y_ts = [3, 1.81, 0.80, 0.25, -0.19, 0.4, 1.24, 1.68, 2.32, 5.05]

    poly_regression(X_tr, y_tr, X_ts, y_ts, max_degree=6, lam=0.0)
    poly_regression(X_tr, y_tr, X_ts, y_ts, max_degree=6, lam=1.0)


# ══════════════════════════════════════════════════════════════
# 3.  GRADIENT DESCENT  (plug-in any objective + gradient)
# ══════════════════════════════════════════════════════════════
def gradient_descent(g_func, grad_func, x0, eta=0.4, num_steps=10, verbose=True):
    """
    Generic gradient descent for a scalar function g(x).

    Parameters
    ----------
    g_func    : callable  — objective, e.g. lambda x: x**2
    grad_func : callable  — gradient,  e.g. lambda x: 2*x
    x0        : float / np.array — starting point
    eta       : learning rate
    num_steps : number of iterations
    verbose   : print each step

    Returns
    -------
    x_final (float), history list of (x, g(x))

    Usage
    -----
    x_opt, hist = gradient_descent(
        g_func    = lambda x: x**2,
        grad_func = lambda x: 2*x,
        x0=1.0, eta=0.4, num_steps=10
    )
    """
    x = x0
    history = []
    if verbose:
        print(f"\n=== Gradient Descent  (eta={eta}, steps={num_steps}) ===")
        print(f"  Initial x: {x}")

    for i in range(num_steps):
        x = x - eta * grad_func(x)
        gx = g_func(x)
        history.append((x, gx))
        if verbose:
            print(f"  Iter {i+1:3d}: x = {x:.6f}  g(x) = {gx:.6f}")

    if verbose:
        print(f"  Optimal x ≈ {x}")
    return x, history


# ─── EXAMPLE ──────────────────────────────────────────────────
if __name__ == "__main__":
    gradient_descent(
        g_func    = lambda x: x**2,
        grad_func = lambda x: 2*x,
        x0=1.0, eta=0.4, num_steps=10
    )


# ══════════════════════════════════════════════════════════════
# 4.  DECISION TREE — impurity helpers (manual calculation)
# ══════════════════════════════════════════════════════════════
def gini(y):
    """Gini impurity of label array y."""
    y = np.array(y)
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs**2)

def entropy(y):
    """Entropy (bits) of label array y."""
    y = np.array(y)
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

def mse_impurity(y):
    """MSE impurity (variance) — used in regression trees."""
    y = np.array(y, dtype=float)
    return float(np.mean((y - np.mean(y))**2))

def weighted_impurity(y_left, y_right, metric="gini"):
    """
    Weighted impurity after a split.
    metric: 'gini' | 'entropy' | 'mse'
    """
    fn = {"gini": gini, "entropy": entropy, "mse": mse_impurity}[metric]
    n = len(y_left) + len(y_right)
    return (len(y_left) / n) * fn(y_left) + (len(y_right) / n) * fn(y_right)

def information_gain(y_parent, y_left, y_right, metric="entropy"):
    """
    IG = impurity(parent) - weighted_impurity(children).
    metric: 'gini' | 'entropy' | 'mse'
    """
    fn = {"gini": gini, "entropy": entropy, "mse": mse_impurity}[metric]
    return fn(y_parent) - weighted_impurity(y_left, y_right, metric)

def find_best_split_1d(X, y, metric="mse"):
    """
    Brute-force best split for a 1-D feature array.
    Returns best_threshold, best_gain, (y_left, y_right).

    Usage
    -----
    thresh, gain, (yl, yr) = find_best_split_1d(X, y, metric="entropy")
    """
    X = np.array(X, dtype=float)
    y = np.array(y)
    fn = {"gini": gini, "entropy": entropy, "mse": mse_impurity}[metric]

    sort_idx = np.argsort(X)
    X_s, y_s = X[sort_idx], y[sort_idx]

    best_thresh = None
    best_gain   = -np.inf

    parent_impurity = fn(y)

    for i in range(len(X_s) - 1):
        # midpoint between consecutive sorted values
        thresh = (X_s[i] + X_s[i+1]) / 2
        mask   = X <= thresh
        yl, yr = y[mask], y[~mask]
        if len(yl) == 0 or len(yr) == 0:
            continue
        gain = parent_impurity - weighted_impurity(yl, yr, metric)
        if gain > best_gain:
            best_gain, best_thresh = gain, thresh
            best_left,  best_right = yl, yr

    print(f"  Best threshold = {best_thresh:.4f}  |  gain ({metric}) = {best_gain:.6f}")
    print(f"  Left  node: n={len(best_left)},  mean/mode = {np.mean(best_left):.4f}")
    print(f"  Right node: n={len(best_right)}, mean/mode = {np.mean(best_right):.4f}")
    return best_thresh, best_gain, (best_left, best_right)

def node_metrics(y, label="Node"):
    """Print gini, entropy and mse for a node."""
    y = np.array(y)
    print(f"\n=== {label} ===")
    print(f"  n        = {len(y)}")
    print(f"  Gini     = {gini(y):.6f}")
    print(f"  Entropy  = {entropy(y):.6f}")
    print(f"  MSE/Var  = {mse_impurity(y):.6f}")
    if np.issubdtype(y.dtype, np.integer) or y.dtype == object:
        vals, cnts = np.unique(y, return_counts=True)
        for v, c in zip(vals, cnts):
            print(f"  class {v}: {c}/{len(y)} = {c/len(y):.4f}")
    else:
        print(f"  mean     = {np.mean(y):.4f}")


# ─── EXAMPLE ──────────────────────────────────────────────────
if __name__ == "__main__":
    y_node = np.array([1, 1, 0, 0, 1, 0])
    node_metrics(y_node, "Root")

    X_feat = np.array([1, 2, 3, 7, 8, 9], dtype=float)
    print("\n=== Best Split ===")
    find_best_split_1d(X_feat, y_node, metric="entropy")


# ══════════════════════════════════════════════════════════════
# 5.  SKLEARN DECISION TREE  (regression & classification)
# ══════════════════════════════════════════════════════════════
def run_decision_tree(X_train, y_train, X_test=None, y_test=None,
                      task="classification",   # "classification" | "regression"
                      criterion=None,          # None → auto-pick
                      max_depth=4,
                      feature_names=None,
                      class_names=None):
    """
    Fits a DecisionTree and prints metrics.

    Returns the fitted model.

    Usage — classification
    -----
    model = run_decision_tree(X_train, y_train, X_test, y_test,
                              task="classification", max_depth=4,
                              feature_names=["f1","f2"],
                              class_names=["No","Yes"])

    Usage — regression
    -----
    model = run_decision_tree(X_train, y_train, X_test, y_test,
                              task="regression", max_depth=3)
    """
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn import metrics

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    if task == "classification":
        crit = criterion or "entropy"
        model = DecisionTreeClassifier(criterion=crit, max_depth=max_depth,
                                       random_state=42)
    else:
        crit = criterion or "squared_error"
        model = DecisionTreeRegressor(criterion=crit, max_depth=max_depth,
                                      random_state=42)

    model.fit(X_train, y_train)
    y_tr_pred = model.predict(X_train)

    print(f"\n=== Decision Tree ({task}, depth={max_depth}, criterion={crit}) ===")

    if task == "classification":
        print(f"  Train accuracy : {metrics.accuracy_score(y_train, y_tr_pred):.4f}")
        if X_test is not None and y_test is not None:
            y_ts_pred = model.predict(np.array(X_test))
            print(f"  Test  accuracy : {metrics.accuracy_score(y_test, y_ts_pred):.4f}")
            print("\n  Confusion Matrix (test):")
            print(metrics.confusion_matrix(y_test, y_ts_pred))
    else:
        mse_tr = metrics.mean_squared_error(y_train, y_tr_pred)
        print(f"  Train MSE : {mse_tr:.6f}")
        if X_test is not None and y_test is not None:
            y_ts_pred = model.predict(np.array(X_test))
            mse_ts = metrics.mean_squared_error(y_test, y_ts_pred)
            print(f"  Test  MSE : {mse_ts:.6f}")

    print("\n  Node impurities (sklearn):")
    print( "  ", model.tree_.impurity)

    return model


# ─── EXAMPLE ──────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    Xtr, Xts, ytr, yts = train_test_split(iris.data, iris.target,
                                           test_size=0.2, random_state=42)
    run_decision_tree(Xtr, ytr, Xts, yts,
                      task="classification", max_depth=4,
                      feature_names=list(iris.feature_names),
                      class_names=list(iris.target_names))


# ══════════════════════════════════════════════════════════════
# 6.  RANDOM FOREST  (classification & regression)
# ══════════════════════════════════════════════════════════════
def run_random_forest(X_train, y_train, X_test=None, y_test=None,
                      task="classification",
                      n_estimators=100, max_depth=None,
                      criterion="gini"):          # "gini"|"entropy" for clf; "squared_error" for reg
    """
    Fits a Random Forest, prints metrics.
    Returns fitted model.

    Usage
    -----
    model = run_random_forest(X_train, y_train, X_test, y_test,
                              task="classification",
                              n_estimators=100, max_depth=5)
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn import metrics

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    if task == "classification":
        crit = criterion if criterion in ("gini", "entropy") else "gini"
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       criterion=crit,
                                       random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      random_state=42)

    model.fit(X_train, y_train)
    y_tr_pred = model.predict(X_train)

    print(f"\n=== Random Forest ({task}, n_estimators={n_estimators}, depth={max_depth}) ===")

    if task == "classification":
        print(f"  Train accuracy : {metrics.accuracy_score(y_train, y_tr_pred):.4f}")
        if X_test is not None and y_test is not None:
            y_ts_pred = model.predict(np.array(X_test))
            print(f"  Test  accuracy : {metrics.accuracy_score(y_test, y_ts_pred):.4f}")
            print("\n  Confusion Matrix (test):")
            print(metrics.confusion_matrix(y_test, y_ts_pred))
    else:
        print(f"  Train MSE : {metrics.mean_squared_error(y_train, y_tr_pred):.6f}")
        if X_test is not None and y_test is not None:
            y_ts_pred = model.predict(np.array(X_test))
            print(f"  Test  MSE : {metrics.mean_squared_error(y_test, y_ts_pred):.6f}")

    return model


# ══════════════════════════════════════════════════════════════
# 7.  MODEL EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════
def print_classification_metrics(y_true, y_pred, class_names=None):
    """
    Prints accuracy, misclassification rate, confusion matrix,
    precision, recall, F1.

    Usage
    -----
    print_classification_metrics(y_test, y_pred, class_names=["cat","dog"])
    """
    from sklearn import metrics
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc  = metrics.accuracy_score(y_true, y_pred)
    cm   = metrics.confusion_matrix(y_true, y_pred)

    print("\n=== Classification Metrics ===")
    print(f"  Accuracy             : {acc:.4f}")
    print(f"  Misclassification    : {1-acc:.4f}")
    print("\n  Confusion Matrix:")
    print(cm)
    print("\n  Per-class report:")
    print(metrics.classification_report(y_true, y_pred,
                                        target_names=class_names))


def print_regression_metrics(y_true, y_pred):
    """
    Prints MSE, RMSE, MAE, R².

    Usage
    -----
    print_regression_metrics(y_test, y_pred)
    """
    from sklearn import metrics
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)

    mse_val  = metrics.mean_squared_error(y_true, y_pred)
    mae_val  = metrics.mean_absolute_error(y_true, y_pred)
    r2_val   = metrics.r2_score(y_true, y_pred)

    print("\n=== Regression Metrics ===")
    print(f"  MSE  : {mse_val:.6f}")
    print(f"  RMSE : {np.sqrt(mse_val):.6f}")
    print(f"  MAE  : {mae_val:.6f}")
    print(f"  R²   : {r2_val:.6f}")


def grid_search_rf(X_trainval, y_trainval,
                   param_grid=None, cv=4, task="classification"):
    """
    Grid-searches a Random Forest over max_depth × n_estimators.
    Returns best estimator + best params.

    Usage
    -----
    best_model, params = grid_search_rf(X_trainval, y_trainval, cv=4)
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    if param_grid is None:
        param_grid = {
            "max_depth":    [1, 5, 10, None],
            "n_estimators": [1, 10, 100, 1000]
        }

    base = (RandomForestClassifier(criterion="gini", random_state=0)
            if task == "classification"
            else RandomForestRegressor(random_state=0))

    scoring = "accuracy" if task == "classification" else "neg_mean_squared_error"

    clf = GridSearchCV(estimator=base, param_grid=param_grid,
                       cv=cv, refit=True, scoring=scoring,
                       return_train_score=True, verbose=0)
    clf.fit(X_trainval, y_trainval)

    print("\n=== Grid Search Results ===")
    print(f"  Best params   : {clf.best_params_}")
    print(f"  Best CV score : {clf.best_score_:.4f}")
    return clf.best_estimator_, clf.best_params_


# ══════════════════════════════════════════════════════════════
# 8.  K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════
def kmeans_step(data, k, centroids, tolerance=1e-4):
    """One iteration of k-means. Returns new_centroids, labels, converged."""
    data      = np.array(data,      dtype=float)
    centroids = np.array(centroids, dtype=float)

    # distances: (k, n)
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    labels    = np.argmin(distances, axis=0)

    new_centroids = np.array([data[labels == j].mean(axis=0)
                               for j in range(k)])
    converged = np.linalg.norm(new_centroids - centroids) < tolerance
    return new_centroids, labels, converged


def kmeans(data, k, init_centroids=None, max_iteration=1000,
           tolerance=1e-4, random_state=42):
    """
    Full k-means.

    Parameters
    ----------
    data           : (n_samples, n_features) array
    k              : number of clusters
    init_centroids : (k, n_features) array, or None for random Forgy init
    max_iteration  : max iterations
    tolerance      : convergence threshold
    random_state   : seed for random init

    Returns
    -------
    centroids (k, n_features), labels (n_samples,), wcss_history

    Usage
    -----
    centroids, labels, wcss = kmeans(data, k=3)
    """
    data = np.array(data, dtype=float)
    rng  = np.random.RandomState(random_state)

    if init_centroids is None:
        idx = rng.choice(len(data), k, replace=False)
        centroids = data[idx].copy()
    else:
        centroids = np.array(init_centroids, dtype=float)

    wcss_history = []
    for i in range(max_iteration):
        centroids, labels, converged = kmeans_step(data, k, centroids, tolerance)
        wcss = sum(np.sum((data[labels == j] - centroids[j])**2)
                   for j in range(k))
        wcss_history.append(wcss)
        if converged:
            print(f"  K-Means converged at iteration {i+1}  |  WCSS = {wcss:.4f}")
            break

    print(f"\n=== K-Means (k={k}) ===")
    print(f"  Final centroids:\n{centroids}")
    cluster_sizes = [int((labels == j).sum()) for j in range(k)]
    print(f"  Cluster sizes  : {cluster_sizes}")
    return centroids, labels, wcss_history


def wcss_elbow(data, k_range=range(1, 9), max_iteration=1000,
               tolerance=1e-4, random_state=42):
    """
    Computes WCSS for each k and prints an elbow-helper table.

    Usage
    -----
    wcss_elbow(data, k_range=range(1, 9))
    """
    data = np.array(data, dtype=float)
    print("\n=== K-Means Elbow Analysis ===")
    print(f"  {'k':>4}  {'Final WCSS':>14}")
    for k in k_range:
        _, _, hist = kmeans(data, k=k, max_iteration=max_iteration,
                            tolerance=tolerance, random_state=random_state)
        print(f"  {k:>4}  {hist[-1]:>14.4f}")


# ─── EXAMPLE ──────────────────────────────────────────────────
if __name__ == "__main__":
    rng_demo = np.random.RandomState(0)
    demo_data = np.vstack([
        rng_demo.randn(30, 2) + [0, 0],
        rng_demo.randn(30, 2) + [5, 5],
        rng_demo.randn(30, 2) + [0, 5],
    ])
    centroids, labels, hist = kmeans(demo_data, k=3)


# ══════════════════════════════════════════════════════════════
# QUICK-REFERENCE CHEAT SHEET  (printed when run as __main__)
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    FUNCTION QUICK REFERENCE                  ║
╠══════════════════════════════════════════════════════════════╣
║  pearson_correlation(x, y)                                   ║
║  pearson_all_features({"f1": [...], "f2": [...]}, target_y)  ║
║                                                              ║
║  poly_regression(X_tr, y_tr, X_ts, y_ts,                    ║
║                  max_degree=6, lam=0.0)                      ║
║                                                              ║
║  gradient_descent(g_func, grad_func, x0,                     ║
║                   eta=0.4, num_steps=10)                     ║
║                                                              ║
║  node_metrics(y, label="Node")                               ║
║  gini(y) / entropy(y) / mse_impurity(y)                      ║
║  weighted_impurity(y_left, y_right, metric="gini")           ║
║  information_gain(y_parent, y_left, y_right, metric="entropy")║
║  find_best_split_1d(X, y, metric="mse")                      ║
║                                                              ║
║  run_decision_tree(X_tr, y_tr, X_ts, y_ts,                   ║
║      task="classification"|"regression", max_depth=4)        ║
║                                                              ║
║  run_random_forest(X_tr, y_tr, X_ts, y_ts,                   ║
║      task="classification", n_estimators=100)                ║
║                                                              ║
║  print_classification_metrics(y_true, y_pred)                ║
║  print_regression_metrics(y_true, y_pred)                    ║
║  grid_search_rf(X_trainval, y_trainval, cv=4)                ║
║                                                              ║
║  kmeans(data, k=3)                                           ║
║  wcss_elbow(data, k_range=range(1,9))                        ║
╚══════════════════════════════════════════════════════════════╝
""")