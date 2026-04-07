"""Stage-1 coarse routing model (pruned decision tree).

Decision tree that maps full feature sets into 1 of N overarching
disease categories (e.g. 'Respiratory', 'Metabolic_Endocrine').
Uses Cost Complexity Pruning.
"""

from typing import Optional, List, Tuple
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


class Stage1Router:
    """Decision Tree router for classifying broad disease categories.

    Uses CCP (Cost-Complexity Pruning) by sweeping alphas.

    Parameters
    ----------
    max_depth : int, default=8
        Maximum depth for interpretability.
    cv_folds : int, default=3
        Folds to use during the CCP alpha sweep cross-validation.
    scoring : str, default='f1_macro'
        Scoring metric for selecting the best tree.
    n_alphas : int, default=20
        Number of alphas to sweep.
    random_state : int, default=42
    """

    def __init__(self, max_depth: int = 8, cv_folds: int = 3,
                 scoring: str = "f1_macro", n_alphas: int = 20,
                 random_state: int = 42):
        self.max_depth = max_depth
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_alphas = n_alphas
        self.random_state = random_state

        self.best_alpha_: Optional[float] = None
        self.model_: Optional[DecisionTreeClassifier] = None

    def fit(self, X: np.ndarray, y_category: np.ndarray) -> "Stage1Router":
        """Fit the decision tree and prune it.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y_category : np.ndarray
            Category integer labels (NOT disease labels).
        """
        # Step 1: Base Tree to extract pruning path
        base_clf = DecisionTreeClassifier(
            max_depth=self.max_depth, random_state=self.random_state)

        path = base_clf.cost_complexity_pruning_path(X, y_category)
        ccp_alphas = path.ccp_alphas

        # Remove negative or near-zero alphas for stability, and sample logarithmically
        alphas = ccp_alphas[ccp_alphas > 0]
        if len(alphas) == 0:
            alphas = np.array([0.0])
        elif len(alphas) > self.n_alphas:
            # Sweep logarithmically across active range
            alphas = np.logspace(
                np.log10(alphas.min()), np.log10(alphas.max()), num=self.n_alphas
            )

        best_score = -1.0
        best_alpha = alphas[0]

        # Step 2: Cross Validation sweep
        for alpha in alphas:
            clf = DecisionTreeClassifier(
                max_depth=self.max_depth,
                ccp_alpha=alpha,
                random_state=self.random_state
            )
            # Make sure we have enough unique classes for splitting across folds
            if len(np.unique(y_category)) > 1:
                scores = cross_val_score(clf, X, y_category, cv=self.cv_folds, scoring=self.scoring)
                mean_score = scores.mean()
            else:
                mean_score = 1.0  # degenerate case

            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha

        # Step 3: Train final model
        self.best_alpha_ = best_alpha
        self.model_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            ccp_alpha=best_alpha,
            random_state=self.random_state
        )
        self.model_.fit(X, y_category)
        print(f"[Stage-1 Router] CCP sweep finished. Best alpha={best_alpha:.5f}, "
              f"Est. F1={best_score:.3f}, Nodes={self.model_.tree_.node_count}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def get_decision_path(self, X_single: np.ndarray, feature_names: List[str]) -> str:
        """Extract a readable IF-THEN rule predicting the routed category."""
        node_indicator = self.model_.decision_path(X_single)
        leaf_id = self.model_.apply(X_single)[0]

        node_indices = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

        rules = []
        for node_id in node_indices:
            if node_id == leaf_id:
                break

            feature_idx = self.model_.tree_.feature[node_id]
            threshold = self.model_.tree_.threshold[node_id]

            feature_name = feature_names[feature_idx]
            patient_val = X_single[0, feature_idx]

            if patient_val <= threshold:
                rules.append(f"{feature_name} <= {threshold:.2f}")
            else:
                rules.append(f"{feature_name} > {threshold:.2f}")

        return " AND ".join(rules) if rules else "No path"
