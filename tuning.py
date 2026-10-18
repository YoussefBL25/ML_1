from cross_validation import cross_validate


def grid_search(y, X, train_fn, score_fn, gammas, lambdas=(0.0,), max_iters=200,
                k=5, seed=0, tune_threshold=True):
    """Grid search over gamma and lambda_. Returns (best, results) as dicts."""
    results, best = [], None
    for gamma in gammas:
        for lambda_ in lambdas:
            params = {"gamma": gamma, "lambda_": lambda_, "max_iters": max_iters}
            threshold, mean_f1, std_f1 = cross_validate(
                y, X, train_fn, score_fn, params, k, seed, tune_threshold)
            row = {"params": params, "threshold": threshold, "f1": mean_f1, "std": std_f1}
            results.append(row)
            print(f"gamma={gamma}, lambda_={lambda_}: "
                  f"F1={mean_f1:.4f} (+/- {std_f1:.4f}), threshold={threshold:.2f}")
            if best is None or mean_f1 > best["f1"]:
                best = row
    return best, results