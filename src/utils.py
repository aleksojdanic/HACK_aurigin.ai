

from pathlib import Path
import numpy as np
import librosa
import pandas as pd
import json

def make_labels_from_csv(
    files: list[Path],
    csv_path: Path,
    filename_col: str = "filename",
    label_col: str = "label",
) -> np.ndarray:
    """
    Erzeugt ein Label-Array für die gegebene Liste von Dateien
    anhand einer CSV mit Spalten z.B. 'filename' und 'label'.

    files: Liste von Path-Objekten (z.B. train_files)
    csv_path: Pfad zur CSV-Datei mit den Labels
    """
    df = pd.read_csv(csv_path)

    # Map von filename -> label bauen
    label_map = dict(zip(df[filename_col], df[label_col]))

    # Für jede Datei das passende Label holen (über p.name = Dateiname ohne Pfad)
    labels = np.array([label_map[p.name] for p in files], dtype=float)

    return labels


def make_labels_from_json(files: list[Path], json_path: Path) -> np.ndarray:
    """
    Lädt Labels aus einer JSON-Datei.
    Erwartetes JSON-Format:
    {
        "file1.wav": 0,
        "file2.wav": 1,
        "file3.wav": 0
    }

    Gibt ein Label-Array in derselben Reihenfolge zurück wie 'files'.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Präfixfreie Zuordnung: filename -> label
    # Reihenfolge: exakt wie in 'files'
    labels = np.array([float(data[p.name]) for p in files])

    return labels




def extract_features_from_file(file_path: Path) -> np.ndarray:
    """
    Lädt eine Audio-Datei und gibt einen einfachen Feature-Vektor zurück.
    Erstmal nur: MFCC-Mittelwerte.
    """
    # Audio laden (mono, feste Samplingrate)
    y, sr = librosa.load(file_path, sr=22050, mono=True)

    # MFCCs berechnen: Matrix der Form (n_mfcc, T)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Über die Zeitachse mitteln → fester Vektor (13,)
    mfcc_mean = mfcc.mean(axis=1)

    return mfcc_mean

def extract_features_for_files(file_paths: list[Path]) -> np.ndarray:
    """
    Nimmt eine Liste von Datei-Pfaden
    und gibt eine Feature-Matrix zurück mit Shape (n_files, n_features).
    """
    feature_list = []

    for file_path in file_paths:
        feats = extract_features_from_file(file_path)
        feature_list.append(feats)

    X = np.vstack(feature_list)  # (N, F)
    return X


def load_data(train_dir: Path, test_dir: Path):
    """
    Lädt alle .wav-Dateien aus train_dir und test_dir,
    extrahiert Features und gibt X_train, y_train, X_test, y_test zurück.
    Labels sind vorerst Dummy-Labels (0.0), bis wir morgen das echte Label-Format kennen.
    """

    # 1) Dateiliste erstellen
    train_files = sorted(list(train_dir.glob("*.wav")))
    test_files = sorted(list(test_dir.glob("*.wav")))

    print(f"Gefundene Trainingsdateien: {len(train_files)}")
    print(f"Gefundene Testdateien: {len(test_files)}")

    # 2) Features extrahieren
    X_train = extract_features_for_files(train_files)
    X_test = extract_features_for_files(test_files)

    # 3) Dummy-Labels (morgen ersetzen wir das durch echte)
    
    y_train = np.random.randint(0, 2, size=len(train_files)).astype(float)
    y_test = np.random.randint(0, 2, size=len(test_files)).astype(float)


    return X_train, y_train, X_test, y_test

