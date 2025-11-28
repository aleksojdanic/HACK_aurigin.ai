import numpy as np
import librosa
from datasets import load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

def extract_mfcc(path, sr=16000, n_mfcc=40):
    """
    Load audio file and extract MFCC features.
    Returns a 1D vector (flattened MFCC).
    """
    try:
        y, sr = librosa.load(path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc.flatten()
    except Exception as e:
        print(f"[ERROR] Failed extracting MFCC from {path}: {e}")
        return np.zeros(n_mfcc * 100)  # fallback: fixed-size dummy vector


def extract_features_from_file(file_dict, n_mfcc=40):
    """
    For a HF dataset entry {'audio': { ... }, 'label': int}
    returns (feature_vector, label)
    """
    try:
        audio = file_dict["audio"]
        path = audio["path"]
        label = file_dict["label"]
        feat = extract_mfcc(path, n_mfcc=n_mfcc)
        return feat, label
    except Exception as e:
        print(f"[ERROR] Could not extract features: {e}")
        return None


def extract_features_for_files(dataset, n_mfcc=40):
    """
    Runs extract_features_from_file() over a HF Dataset.
    Returns X, y numpy arrays.
    """
    X = []
    y = []

    for item in dataset:
        res = extract_features_from_file(item, n_mfcc=n_mfcc)
        if res is None:
            continue
        feat, label = res
        X.append(feat)
        y.append(label)

    return np.array(X), np.array(y)


def load_data_hf(limit_train=None, limit_val=None, limit_test=None):
    """
    Loads HF dataset and returns feature-extracted arrays.
    Supports optional limits for debugging.
    """

    try:
        dataset = load_dataset("aurigin/tvm_dataset", cache_dir="hf_cache")
    except HfHubHTTPError as e:
        print("[ERROR] HF dataset not accessible. Check your HF token.")
        raise e

    train = dataset["train"]
    val = dataset["validation"]
    test = dataset["test"]

    # Apply limits
    if limit_train:
        train = train.select(range(min(limit_train, len(train))))
    if limit_val:
        val = val.select(range(min(limit_val, len(val))))
    if limit_test:
        test = test.select(range(min(limit_test, len(test))))

    print(f"Loaded HF Dataset: train={len(train)}, val={len(val)}, test={len(test)}")

    X_train, y_train = extract_features_for_files(train)
    X_val, y_val = extract_features_for_files(val)
    X_test, test_split = extract_features_for_files(test)

    return X_train, y_train, X_val, y_val, X_test, test_split
