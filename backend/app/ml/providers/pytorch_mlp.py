from __future__ import annotations

import time
from typing import Any

import numpy as np

from app.ml.providers.base import SuggestionProvider, TrainConfig, TrainReport


class PyTorchMLPProvider(SuggestionProvider):
    id = "pytorch_mlp"
    display_name = "PyTorch MLP"

    def __init__(self) -> None:
        self._model = None
        self._classes: np.ndarray | None = None
        self._torch = None

    def config_schema(self) -> dict[str, Any]:
        return {
            "epochs": {"type": "integer", "default": 50, "minimum": 1},
            "lr": {"type": "number", "default": 0.001},
            "batch_size": {"type": "integer", "default": 32},
            "hidden": {"type": "array", "default": [128, 64]},
        }

    def _import_torch(self):
        if self._torch is None:
            try:
                import torch
            except ImportError as e:
                raise RuntimeError(
                    "PyTorch is not installed. Install torch or choose another provider."
                ) from e
            self._torch = torch
        return self._torch

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def fit(self, X: np.ndarray, y: np.ndarray, *, config: TrainConfig) -> TrainReport:
        torch = self._import_torch()
        t0 = time.perf_counter()
        pc = config.provider_config
        epochs = int(pc.get("epochs", 50))
        lr = float(pc.get("lr", 0.001))
        batch_size = int(pc.get("batch_size", 32))
        hidden = [int(x) for x in pc.get("hidden", [128, 64])]

        classes = np.unique(y)
        self._classes = classes
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([class_to_idx[v] for v in y], dtype=np.int64)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_idx, dtype=torch.long)

        n_in = X.shape[1]
        n_out = len(classes)
        layers: list[Any] = []
        prev = n_in
        for h in hidden:
            layers.extend([torch.nn.Linear(prev, h), torch.nn.ReLU()])
            prev = h
        layers.append(torch.nn.Linear(prev, n_out))
        net = torch.nn.Sequential(*layers)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        net.train()
        n = len(y)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb, yb = X_t[idx], y_t[idx]
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                opt.step()

        self._model = net
        ms = (time.perf_counter() - t0) * 1000
        acc = self._simple_holdout_accuracy(X, y, hidden, epochs, lr, batch_size, config.random_state)
        return TrainReport(success=True, message="PyTorch MLP trained", accuracy=acc, duration_ms=ms)

    def _simple_holdout_accuracy(
        self, X, y, hidden, epochs, lr, batch_size, random_state
    ) -> float | None:
        if len(y) < 10:
            return None
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(len(y))
        split = int(0.8 * len(y))
        tr, te = idx[:split], idx[split:]
        if len(te) == 0:
            return None
        sub = PyTorchMLPProvider()
        sub.fit(
            X[tr],
            y[tr],
            config=TrainConfig(
                provider_config={
                    "epochs": max(5, epochs // 5),
                    "lr": lr,
                    "batch_size": batch_size,
                    "hidden": hidden,
                }
            ),
        )
        pred = sub.predict(X[te])
        return float(np.mean(pred == y[te]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        torch = self._import_torch()
        if self._model is None or self._classes is None:
            raise RuntimeError("Model not trained")
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.tensor(X, dtype=torch.float32))
            idx = logits.argmax(dim=1).numpy()
        return self._classes[idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        torch = self._import_torch()
        if self._model is None or self._classes is None:
            return None
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.tensor(X, dtype=torch.float32))
            proba = torch.softmax(logits, dim=1).numpy()
        return proba
