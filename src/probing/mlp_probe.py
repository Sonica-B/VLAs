"""
MLP probe (single hidden layer, 512 units) for physics property prediction.

Complements the linear probe — if MLP R² >> linear probe R², physics information
is encoded non-linearly in the activations. If roughly equal, linear structure
is sufficient, suggesting the physics encoding is linearly decodable.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset


class MLPProbeNetwork(nn.Module):
    """Single hidden-layer MLP: D → 512 → 1.

    Args:
        input_dim: Feature dimension D.
        hidden_dim: Hidden layer width. Default 512.
        dropout: Dropout probability. Default 0.1.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, D] activation features.

        Returns:
            [B, 1] predicted physics property value.
        """
        return self.net(x)


class MLPProbe:
    """Training wrapper around MLPProbeNetwork.

    Args:
        input_dim: Activation feature dimension.
        hidden_dim: MLP hidden layer width. Default 512.
        dropout: Dropout probability. Default 0.1.
        lr: Adam learning rate. Default 1e-3.
        weight_decay: Adam weight decay. Default 1e-4.
        epochs: Training epochs. Default 50.
        batch_size: Mini-batch size. Default 512.
        early_stopping_patience: Stop if val loss doesn't improve for N epochs.
        normalize_features: Apply StandardScaler before training. Default True.
        device: Training device. Default "cuda" if available.

    Example:
        >>> probe = MLPProbe(input_dim=1280)
        >>> probe.fit(X_train, y_train, X_val, y_val)
        >>> metrics = probe.score(X_test, y_test)
        >>> print(metrics["r2"])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 512,
        early_stopping_patience: int = 10,
        normalize_features: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.normalize_features = normalize_features
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._network: Optional[MLPProbeNetwork] = None
        self._scaler: Optional[StandardScaler] = StandardScaler() if normalize_features else None
        self._is_fitted = False
        self.training_history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    def _build_network(self) -> MLPProbeNetwork:
        return MLPProbeNetwork(self.input_dim, self.hidden_dim, self.dropout).to(self.device)

    def _filter_nan(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        valid = ~np.isnan(y)
        return X[valid], y[valid]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "MLPProbe":
        """Train the MLP probe.

        Args:
            X_train: [N_train, D] training activations.
            y_train: [N_train] training labels (NaN = background, excluded).
            X_val: Optional [N_val, D] validation activations.
            y_val: Optional [N_val] validation labels.
            verbose: Print training progress every 10 epochs.

        Returns:
            self (for chaining).
        """
        X_train, y_train = self._filter_nan(X_train, y_train)

        if self._scaler is not None:
            X_train = self._scaler.fit_transform(X_train).astype(np.float32)
            if X_val is not None and y_val is not None:
                X_val_f, y_val_f = self._filter_nan(X_val, y_val)
                X_val_f = self._scaler.transform(X_val_f).astype(np.float32)
        else:
            X_train = X_train.astype(np.float32)
            if X_val is not None and y_val is not None:
                X_val_f, y_val_f = self._filter_nan(X_val, y_val)

        # Build dataset
        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        self._network = self._build_network()
        optimizer = optim.Adam(
            self._network.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_count = 0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            # Training
            self._network.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                pred = self._network(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            train_loss = epoch_loss / len(train_loader)
            self.training_history["train_loss"].append(train_loss)

            # Validation
            if X_val is not None:
                val_loss = self._compute_val_loss(X_val_f, y_val_f, criterion)
                self.training_history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_count = 0
                    best_state = {k: v.clone() for k, v in self._network.state_dict().items()}
                else:
                    patience_count += 1
                    if patience_count >= self.early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break

            if verbose and epoch % 10 == 0:
                val_str = f"  val_loss={val_loss:.4f}" if X_val is not None else ""
                print(f"  Epoch {epoch:3d}/{self.epochs}: train_loss={train_loss:.4f}{val_str}")

        # Restore best weights
        if best_state is not None:
            self._network.load_state_dict(best_state)

        self._is_fitted = True
        return self

    @torch.no_grad()
    def _compute_val_loss(
        self, X_val: np.ndarray, y_val: np.ndarray, criterion: nn.Module
    ) -> float:
        self._network.eval()
        X_t = torch.from_numpy(X_val).to(self.device)
        y_t = torch.from_numpy(y_val.astype(np.float32)).unsqueeze(1).to(self.device)
        pred = self._network(X_t)
        return criterion(pred, y_t).item()

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict physics property values.

        Args:
            X: [N, D] activation features.

        Returns:
            [N] predicted values, float32.
        """
        self._check_fitted()
        if self._scaler is not None:
            X = self._scaler.transform(X).astype(np.float32)
        else:
            X = X.astype(np.float32)
        self._network.eval()
        X_t = torch.from_numpy(X).to(self.device)
        preds = self._network(X_t).cpu().numpy().squeeze(1)
        return preds.astype(np.float32)

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics on test data.

        Args:
            X: [N_test, D] activation features.
            y: [N_test] ground truth labels (may contain NaN).

        Returns:
            Dict with keys: r2, pearson_r, mse, mae.
        """
        self._check_fitted()
        valid = ~np.isnan(y)
        X_valid, y_valid = X[valid], y[valid]
        y_pred = self.predict(X_valid)

        r2 = r2_score(y_valid, y_pred)
        pearson_r, _ = pearsonr(y_valid, y_pred) if len(y_valid) > 2 else (0.0, 1.0)
        mse = float(np.mean((y_valid - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_valid - y_pred)))

        return {"r2": float(r2), "pearson_r": float(pearson_r), "mse": mse, "mae": mae}

    @torch.no_grad()
    def get_saliency(self, X: np.ndarray) -> np.ndarray:
        """Compute gradient × input saliency scores for each feature.

        Useful for understanding which activation dimensions drive the prediction.
        Note: this is different from per-patch saliency (which is computed
        by training separate probes per patch).

        Args:
            X: [N, D] activation features.

        Returns:
            [D] mean absolute gradient × input score per feature dimension.
        """
        self._check_fitted()
        if self._scaler is not None:
            X = self._scaler.transform(X).astype(np.float32)
        X_t = torch.from_numpy(X).to(self.device).requires_grad_(True)
        self._network.eval()
        pred = self._network(X_t)
        pred.sum().backward()
        saliency = (X_t.grad * X_t).abs().mean(dim=0).cpu().numpy()
        return saliency.astype(np.float32)

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._network is None:
            raise RuntimeError("MLPProbe has not been fitted. Call fit() first.")
