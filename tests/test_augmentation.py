import pytest
import numpy as np
from src.augmentation.hybrid import HybridAugmentor
from src.augmentation.ctgan_handler import CTGAN_AVAILABLE


def test_hybrid_augmentor_smote_only():
    # Construct a highly imbalanced dataset
    # Class 0: 100 samples, Class 1: 10 samples
    np.random.seed(42)
    # Define 5 categorical cols + 1 continuous col
    X_maj_cat = np.random.randint(0, 2, size=(100, 5))
    X_maj_num = np.random.rand(100, 1)
    X_maj = np.hstack([X_maj_cat, X_maj_num])
    y_maj = np.zeros(100)
    
    X_min_cat = np.random.randint(0, 2, size=(15, 5))
    X_min_num = np.random.rand(15, 1)
    X_min = np.hstack([X_min_cat, X_min_num])
    y_min = np.ones(15)
    
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([y_maj, y_min])
    
    # Only test SMOTE to avoid slow tests via GAN
    aug = HybridAugmentor(
        cat_indices=[0, 1, 2, 3, 4],
        use_smote=True,
        use_ctgan=False,
        use_tomek=False,
        smote_kwargs={"k_neighbors": 3}
    )
    
    X_res, y_res = aug.fit_resample(X, y)
    
    # SMOTE should balance the minority class to match the majority class
    counts = np.bincount(y_res.astype(int))
    assert counts[0] == 100
    assert counts[1] == 100
    assert X_res.shape[0] == 200


@pytest.mark.skipif(not CTGAN_AVAILABLE, reason="ctgan not installed")
def test_hybrid_augmentor_ctgan_injection():
    # If CTGAN is installed, verify the class size inflation
    np.random.seed(42)
    X = np.random.randint(0, 2, size=(50, 2))
    y = np.zeros(50)
    
    aug = HybridAugmentor(
        cat_indices=[0, 1],
        use_smote=False,
        use_ctgan=True,
        use_tomek=False,
        ctgan_kwargs={"epochs": 1}
    )
    
    X_res, y_res = aug.fit_resample(X, y)
    
    # We requested CTGAN to add 10% more records to majority class
    # 50 // 10 = 5 synthetic records
    assert X_res.shape[0] == 55
