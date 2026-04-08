import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class NeuroModel:
    def __init__(self) -> None:
        self.model: SVC | None = None
        self.scaler: StandardScaler | None = None

    def train(self, features_list, labels_list) -> None:
        """Train an SVM classifier with probability estimates enabled."""
        if len(features_list) == 0 or len(labels_list) == 0:
            raise ValueError("features_list and labels_list must not be empty")

        if len(features_list) != len(labels_list):
            raise ValueError("features_list and labels_list must have the same length")

        x_train = np.asarray(features_list, dtype=np.float32)
        y_train = np.asarray(labels_list)

        if x_train.ndim != 2:
            raise ValueError("features_list must be a 2D array-like structure")

        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x_train)

        self.model = SVC(
            probability=True,
            kernel="linear",
            C=1.0,
            class_weight="balanced",
        )
        self.model.fit(x_scaled, y_train)

    def predict(self, feature_vector):
        """Return predicted label and confidence score for a single feature vector."""
        if self.model is None:
            raise ValueError("Model is not trained or loaded")

        x_input = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        if self.scaler is not None:
            x_input = self.scaler.transform(x_input)

        probabilities = self.model.predict_proba(x_input)[0]
        best_idx = int(np.argmax(probabilities))

        predicted_label = str(self.model.classes_[best_idx])
        confidence = float(probabilities[best_idx])
        return predicted_label, confidence

    def save(self, file_path: str | Path) -> None:
        """Save the trained model to a .pkl file."""
        if self.model is None:
            raise ValueError("Model is not trained or loaded")

        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with target.open("wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    def load(self, file_path: str | Path) -> None:
        """Load a model from a .pkl file."""
        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(f"Model file not found: {source}")

        with source.open("rb") as f:
            loaded_model = pickle.load(f)

        # Backward compatibility: older files may contain only an SVC instance.
        if isinstance(loaded_model, SVC):
            self.model = loaded_model
            self.scaler = None
            return

        if not isinstance(loaded_model, dict):
            raise TypeError("Loaded model file has an unsupported format")

        model_obj: Any = loaded_model.get("model")
        scaler_obj: Any = loaded_model.get("scaler")

        if not isinstance(model_obj, SVC):
            raise TypeError("Loaded object does not contain an sklearn.svm.SVC model")

        if scaler_obj is not None and not isinstance(scaler_obj, StandardScaler):
            raise TypeError("Loaded object has an invalid scaler")

        self.model = model_obj
        self.scaler = scaler_obj
