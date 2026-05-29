"""Training entry point for the medical diagnosis pipeline."""

import os
import yaml
import sys
import argparse
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder

# Insert path to allow running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple
from src.pipeline import (
    SymptomCanonicalizer, PassThroughCanonicalizer,
    load_raw_dataset, load_binary_matrix_dataset,
    ThreeStateEncoder, FeatureEngineer,
)
from src.pipeline.feature_eng import (
    BINARY_MATRIX_SYMPTOM_CATEGORIES, BINARY_MATRIX_INTERACTION_PAIRS,
)
# HybridAugmentor (and its heavy imblearn dependency) is imported lazily inside
# the augmentation branch — it's only needed when augmentation.enabled is true,
# so importing this module for normal training doesn't require imblearn.
from src.models.stage1_router import Stage1Router
from src.models.stage2_ensemble import Stage2Ensemble
from src.models.calibrator import ProbabilityCalibrator, compute_ece
from src.models.explainer import PredictionExplainer
from src.monitoring.psi_monitor import PSIMonitor

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def align_disease_categories(y_labels: np.ndarray,
                             label_encoder: LabelEncoder,
                             config: dict) -> Tuple[np.ndarray, LabelEncoder]:
    """Create category integer targets from exact disease labels.

    Every disease present in the data MUST appear in config['disease_categories'].
    The previous behaviour silently routed unmapped diseases into a hard-coded
    'Metabolic_Endocrine' bucket — on the new 100-disease dataset that collapsed
    92 diseases into one category and destroyed Stage-1 routing. We now fail loudly
    instead, so a missing mapping is caught at training time rather than masked.
    """
    disease_to_cat = {}
    cat_names = list(config["disease_categories"].keys())
    cat_to_id = {cat: i for i, cat in enumerate(cat_names)}

    for cat_name, diseases in config["disease_categories"].items():
        for d in diseases:
            disease_to_cat[str(d).strip()] = cat_name

    cat_encoder = LabelEncoder()
    # Use config INSERTION order (not alphabetical) for the id<->name mapping.
    # The router, Stage-2 ensembles, and calibrator are all keyed on cat_to_id
    # (insertion order). LabelEncoder().fit() would sort classes_ alphabetically,
    # desyncing inverse_transform from those trained ids — that mismatch made a
    # correctly-predicted disease display the wrong category name. Setting
    # classes_ directly keeps decode consistent with cat_to_id.
    cat_encoder.classes_ = np.array(cat_names, dtype=object)

    # Validate coverage up-front against the actual classes in the data.
    present_diseases = [str(d).strip() for d in label_encoder.classes_]
    unmapped = sorted(d for d in present_diseases if d not in disease_to_cat)
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} disease(s) in the data have no category in "
            f"config['disease_categories'] and would corrupt Stage-1 routing: "
            f"{unmapped[:10]}{' ...' if len(unmapped) > 10 else ''}. "
            f"Add them to config.yaml under disease_categories."
        )

    y_cat_ids = np.zeros_like(y_labels)
    for i, label_idx in enumerate(y_labels):
        disease_name = str(label_encoder.inverse_transform([label_idx])[0]).strip()
        y_cat_ids[i] = cat_to_id[disease_to_cat[disease_name]]

    return y_cat_ids, cat_encoder


def main() -> None:
    # ── 1. Load config ────────────────────────────────────────────────────────
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    print("=== Medical Diagnosis Pipeline Training ===")

    # ── 2. Data Pipeline ──────────────────────────────────────────────────────
    print("\n[Phase 1] Data Processing")
    data_cfg = config.get("data", {})
    fmt = data_cfg.get("format", "legacy_names")

    # Resolve dataset path relative to the project dir (config.yaml's folder).
    project_dir = os.path.dirname(CONFIG_PATH)
    data_path = os.path.join(project_dir, config["paths"]["dataset"])
    if not os.path.exists(data_path):
        data_path = config["paths"]["dataset"]  # fallback for direct invocation

    print(f"Loading dataset from {data_path} (format={fmt})...")

    if fmt == "binary_matrix":
        # New dataset: wide 0/1 matrix, no token canonicalization needed.
        label_col = data_cfg.get("label_col", "diseases")
        df_raw = load_binary_matrix_dataset(data_path, label_col=label_col)

        print("Canonicalization: pass-through (matrix already binary)...")
        canonicalizer = PassThroughCanonicalizer()
        df_clean = canonicalizer.fit_transform(df_raw)

        print("Encoding binary symptom matrix...")
        encoder = ThreeStateEncoder()
        X, y = encoder.fit_transform_binary_matrix(df_clean)

        print("Engineering layout-derived features...")
        feat_kwargs = config.get("features", {})
        feat_eng = FeatureEngineer(
            encoder.get_feature_names(),
            symptom_categories=BINARY_MATRIX_SYMPTOM_CATEGORIES,
            add_interactions=feat_kwargs.get("add_interactions", False),
            interaction_pairs=BINARY_MATRIX_INTERACTION_PAIRS,
        )
        X_enriched = feat_eng.transform(X)
    else:
        # Legacy dataset.csv: disease + symptom-name columns.
        df_raw = load_raw_dataset(data_path)

        print("Canonicalizing symptoms...")
        canonicalizer = SymptomCanonicalizer()
        df_clean = canonicalizer.fit_transform(df_raw)

        print("Encoding 3-state symptom matrix...")
        encoder = ThreeStateEncoder()
        X, y = encoder.fit_transform(df_clean)

        print("Engineering layout-derived features...")
        feat_kwargs = config.get("features", {})
        feat_eng = FeatureEngineer(
            encoder.get_feature_names(),
            add_interactions=feat_kwargs.get("add_interactions", False),
        )
        X_enriched = feat_eng.transform(X)

    # ── 3. Train-Test Split ───────────────────────────────────────────────────
    print(f"\n[Phase 2/3] Train-Test Split & Augmentation")
    # Convert y targets to Phase-1 categories
    y_categories, cat_encoder = align_disease_categories(y, encoder.label_encoder_, config)

    X_train, X_test, y_cat_train, y_cat_test, y_train, y_test = train_test_split(
        X_enriched, y_categories, y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=categories_stratify_fallback(y_categories, y)
    )

    # Hold out the calibration set NOW, before any augmentation or model fitting,
    # so the ensemble never sees it. Carving calibration data out of X_train
    # *after* fitting (the previous behaviour) leaks training rows into Platt
    # scaling, producing overconfident calibration and a meaningless ECE.
    cal_size = config["training"].get("calibration_size", 0.15)
    X_train, X_cal, y_cat_train, y_cat_cal, y_train, y_cal = train_test_split(
        X_train, y_cat_train, y_train,
        test_size=cal_size,
        random_state=config["training"]["random_state"],
        stratify=y_train
    )

    # ── 4. Augmentation (Optional) ────────────────────────────────────────────
    aug_cfg = config.get("augmentation", {})
    if aug_cfg.get("enabled", False):
        print("Applying hybrid augmentation (SMOTE + CTGAN + Tomek)...")
        # Imported here (not at module top) so the heavy imblearn dependency is
        # only required when augmentation is actually enabled.
        from src.augmentation.hybrid import HybridAugmentor
        # Base features are categorical (1/0), derived features are continuous
        cat_indices = list(range(len(encoder.get_feature_names())))
        augmentor = HybridAugmentor(
            cat_indices=cat_indices,
            use_smote=True,
            use_ctgan=aug_cfg.get("ctgan", {}).get("enabled", False),
            use_tomek=aug_cfg.get("tomek_links", True),
            smote_kwargs=aug_cfg.get("smote"),
            ctgan_kwargs=aug_cfg.get("ctgan")
        )
        X_train_aug, y_train_aug = augmentor.fit_resample(X_train, y_train)
        
        # Realignment of categories
        y_cat_train_aug, _ = align_disease_categories(y_train_aug, encoder.label_encoder_, config)
        
        X_train, y_train, y_cat_train = X_train_aug, y_train_aug, y_cat_train_aug
    else:
        print("Augmentation is disabled (dataset perfectly balanced). Skipping.")

    print("\n[Augmentation] Symptom Dropout (Simulating sparse UI inputs)")
    np.random.seed(config.get("training", {}).get("random_state", 42))

    # Number of base symptom columns (before derived features)
    n_base = len(encoder.get_feature_names())

    # Fraction of genuinely-absent symptoms to encode as -1 ("confirmed absent")
    # so the model actually learns the I-MNAR negative-evidence signal instead of
    # treating -1 identically to 0 (unknown) at inference time.
    neg_rate = aug_cfg.get("negative_evidence_rate", 0.10)

    # Original base matrix: distinguishes symptoms the disease genuinely lacks
    # (==0) from those it presents with (==1). -1 must only ever land on the
    # former — marking a disease's true symptom as "confirmed absent" would
    # teach the model contradictory evidence.
    base_orig = X_train[:, :n_base]

    # Create synthetic copies with varying dropout rates. Tiers are configurable
    # so the large binary-matrix dataset can use fewer copies (it already has
    # realistic sparsity); the legacy dataset keeps the original 4-tier schedule.
    dropout_rates = aug_cfg.get("dropout_tiers", [0.15, 0.30, 0.50, 0.80])
    augmented_X = [X_train]
    augmented_y = [y_train]
    augmented_ycat = [y_cat_train]

    for rate in dropout_rates:
        # Drop out base symptom columns only (derived features are recomputed).
        X_base_drop = base_orig.copy().astype(np.float32)
        drop_mask = np.random.rand(X_base_drop.shape[0], n_base) < rate
        X_base_drop[drop_mask & (X_base_drop == 1)] = 0

        # Negative evidence: flip a fraction of genuinely-absent symptoms to -1.
        # Restricted to originally-absent cells so we never contradict a true
        # symptom, and so the model learns "-1 here argues against diseases that
        # require this symptom."
        if neg_rate > 0:
            absent_orig = base_orig == 0
            neg_mask = (np.random.rand(X_base_drop.shape[0], n_base) < neg_rate) & absent_orig
            X_base_drop[neg_mask] = -1

        # Recompute derived features in a single vectorized pass so category
        # counts and symptom_count stay consistent. transform() takes the base
        # matrix and returns base + derived, matching X_train's column layout.
        # (symptom_count / category counts use (X > 0), so -1 correctly does not
        # count as "present".)
        X_drop = feat_eng.transform(X_base_drop)

        augmented_X.append(X_drop)
        augmented_y.append(y_train)
        augmented_ycat.append(y_cat_train)
        print(f"  Created dropout tier at {rate:.0%} (neg-evidence rate {neg_rate:.0%})")

    X_train = np.vstack(augmented_X)
    y_train = np.concatenate(augmented_y)
    y_cat_train = np.concatenate(augmented_ycat)

    # ── 5. Stage-1 Router ─────────────────────────────────────────────────────
    print("\n[Phase 2] Training Stage-1 Router (Pruned DT)")
    stage1_cfg = config["stage1"]
    router = Stage1Router(
        max_depth=stage1_cfg["max_depth"],
        cv_folds=stage1_cfg["cv_folds"],
        scoring=stage1_cfg.get("scoring", "f1_macro"),
        n_alphas=stage1_cfg.get("ccp_alpha_sweep", 20),
        random_state=config["training"]["random_state"]
    )
    router.fit(X_train, y_cat_train)
    
    # Evaluate router
    y_cat_pred = router.predict(X_test)
    f1_s1 = f1_score(y_cat_test, y_cat_pred, average="macro")
    print(f"Stage-1 Category Routing F1-Score: {f1_s1:.4f}")

    # ── 6. Stage-2 Ensemble ───────────────────────────────────────────────────
    print("\n[Phase 2] Training Stage-2 Fine-Grained Ensembles")
    stage2_cfg = config["stage2"]
    ensemble = Stage2Ensemble(
        dt_kwargs=stage2_cfg["decision_tree"],
        rf_kwargs=stage2_cfg["random_forest"],
        lgb_kwargs=stage2_cfg["lightgbm"],
        weights=stage2_cfg["ensemble_weights"]
    )
    ensemble.fit(X_train, y_train, y_cat_train)

    # ── 7. Calibration + Explainer + PSI ──────────────────────────────────────
    print("\n[Phase 3] Calibration, Explainability & Monitoring")
    
    print("Fitting Platt Calibrator...")
    # Calibration set was held out before augmentation/training (see Phase 2),
    # so X_cal/y_cal/y_cat_cal are genuinely unseen by the ensemble. No re-split.
    print(f"  Calibrating on {len(X_cal)} held-out samples (unseen by ensemble).")

    calibrator = ProbabilityCalibrator()
    calibrator.fit(ensemble, X_cal, y_cal, y_cat_cal)
    
    print("Fitting SHAP Explainers...")
    feature_names = feat_eng.get_all_feature_names()
    explainer = PredictionExplainer(router, ensemble, feature_names)
    # Using small subset of training data as background dataset
    bg_size = min(300, len(X_train))
    idx = np.random.choice(len(X_train), bg_size, replace=False)
    explainer.fit_shap(X_train[idx], y_cat_train[idx])
    
    print("Initializing PSI Monitor...")
    psi_monitor = PSIMonitor(n_bins=config.get("monitoring", {}).get("n_bins", 10))
    psi_monitor.fit(X_train)

    # ── 8. Evaluate Full Pipeline ─────────────────────────────────────────────
    print("\n=== Evaluation (Final Classifications) ===")

    # Route every test row first (vectorized), then run Stage-2 in per-category
    # batches instead of one row at a time. Semantics are identical to the old
    # per-record loop but this scales to the new dataset's ~19k-row test set.
    cats_test = router.predict(X_test)
    n_test = len(X_test)
    y_pred_final = np.zeros(n_test, dtype=int)
    top_conf = np.zeros(n_test, dtype=float)

    for cat_i in np.unique(cats_test):
        rows = np.where(cats_test == cat_i)[0]
        X_rows = X_test[rows]
        try:
            proba = calibrator.predict_proba(X_rows, cat_i)        # (m, n_cls_in_cat)
            preds = ensemble.predict(X_rows, cat_i)                # disease labels
            y_pred_final[rows] = preds
            top_conf[rows] = proba.max(axis=1)
        except Exception as e:
            print(f"  [eval] category {cat_i} fell back ({e})")
            # Degenerate fallback: most common class of the category's ensemble.
            ens = ensemble.ensembles_.get(cat_i)
            fallback = int(ens.classes_[0]) if ens is not None and hasattr(ens, "classes_") else 0
            y_pred_final[rows] = fallback
            top_conf[rows] = 1.0 / max(len(encoder.get_disease_names()), 1)

    # ECE proxy: top-label confidence vs correctness over all test rows.
    correct = (y_pred_final == y_test).astype(int)
    y_proba_cat_all = np.column_stack([1 - top_conf, top_conf])
    ece_score = compute_ece(correct, y_proba_cat_all, n_bins=10)
    print(f"Post-Calibration Expected Calibration Error (ECE): {ece_score:.4f}")

    # Overall disease-level metrics
    macro_f1 = f1_score(y_test, y_pred_final, average="macro")
    acc = (y_pred_final == y_test).mean()
    print(f"Disease-level macro-F1: {macro_f1:.4f} | accuracy: {acc:.4f}")

    # Output metrics
    target_names = encoder.label_encoder_.classes_
    # y_test / y_pred_final need not cover all classes (especially with the
    # degenerate fallback), so pin labels to the full class index range to keep
    # target_names aligned and avoid a ValueError.
    print(classification_report(
        y_test, y_pred_final,
        labels=np.arange(len(target_names)),
        target_names=target_names,
        zero_division=0,
    ))

    # Save artifacts
    print("\nSaving completed models...")
    os.makedirs(config["paths"]["models_dir"], exist_ok=True)
    out_dir = config["paths"]["models_dir"]
    
    joblib.dump(canonicalizer, os.path.join(out_dir, "canonicalizer.pkl"))
    joblib.dump(encoder,       os.path.join(out_dir, "encoder.pkl"))
    joblib.dump(feat_eng,      os.path.join(out_dir, "feat_eng.pkl"))
    joblib.dump(cat_encoder,   os.path.join(out_dir, "cat_encoder.pkl"))
    joblib.dump(router,        os.path.join(out_dir, "stage1.pkl"))
    joblib.dump(ensemble,      os.path.join(out_dir, "stage2.pkl"))
    joblib.dump(calibrator,    os.path.join(out_dir, "calibrator.pkl"))
    joblib.dump(explainer,     os.path.join(out_dir, "explainer.pkl"))
    joblib.dump(psi_monitor,   os.path.join(out_dir, "psi.pkl"))
    
    print(f"All serialized artifacts dumped to {out_dir}/")


def categories_stratify_fallback(y_cat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Helper to stratify primarily by fine-grained disease labels."""
    return y


if __name__ == "__main__":
    main()
