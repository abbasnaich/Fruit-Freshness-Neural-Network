# =============================================================================
#  Fruit Freshness Neural Network
#  Lab 4 - Artificial Neural Network & Deep Learning
#
#  Architecture:
#    Input  : 2 image-based feature vectors (Peach + Strawberry) = 18 features
#    Hidden1: 16 neurons  (ReLU)
#    Hidden2:  8 neurons  (ReLU)
#    Hidden3:  4 neurons  (ReLU)
#    Output :  1 neuron   (Sigmoid – freshness probability)
#
#  Transfer / Activation functions demonstrated on output:
#    - Hard-Limit  (Heaviside step)
#    - Sigmoid     (logistic)
#    - Tanh        (hyperbolic tangent)
# =============================================================================

import os
import numpy as np
from PIL import Image

# ── 1. CONFIGURATION ──────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), "Fruits Dataset")
IMG_SIZE    = (64, 64)   # resize every image before feature extraction
SEED        = 42
np.random.seed(SEED)


# ── 2. IMAGE FEATURE EXTRACTION ───────────────────────────────────────────────
def load_images_from_folder(folder: str) -> list:
    """Load all JPEG/PNG images from *folder* and return a list of RGB arrays."""
    images = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, fname)
            try:
                img = Image.open(path).convert("RGB").resize(IMG_SIZE)
                images.append(np.array(img, dtype=np.float32))
            except Exception:
                pass   # skip unreadable files
    return images


def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Extract a 9-element numerical feature vector from one RGB image.

    Features:
      [0-2] Mean  R, G, B channels        (colour statistics)
      [3-5] Std   R, G, B channels        (texture / colour spread)
      [6]   Overall brightness            (mean greyscale value)
      [7]   Contrast                      (std of greyscale)
      [8]   Red-to-Green ratio            (ripeness indicator)
    """
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    grey    = 0.299 * r + 0.587 * g + 0.114 * b

    features = np.array([
        r.mean(), g.mean(), b.mean(),          # colour means
        r.std(),  g.std(),  b.std(),           # colour std devs
        grey.mean(),                           # brightness
        grey.std(),                            # contrast
        r.mean() / (g.mean() + 1e-8),         # red-to-green ratio
    ], dtype=np.float32)

    return features


def build_dataset(fruit: str) -> tuple:
    """
    Load fresh (label=1) and rotten (label=0) images for *fruit*.
    Returns (X, y) with X.shape == (N, 9).

    Folder naming follows the dataset convention:
      fresh_peaches_done, rotten_peaches_done
      fresh_strawberries_done, rotten_strawberries_done
    """
    # Map singular fruit name to the exact folder suffix used in the dataset
    folder_map = {
        "peach":      "peaches",
        "strawberry": "strawberries",
    }
    plural = folder_map[fruit]

    fresh_dir  = os.path.join(DATASET_DIR, f"fresh_{plural}_done")
    rotten_dir = os.path.join(DATASET_DIR, f"rotten_{plural}_done")

    fresh_imgs  = load_images_from_folder(fresh_dir)
    rotten_imgs = load_images_from_folder(rotten_dir)

    fresh_feats  = np.array([extract_features(img) for img in fresh_imgs])
    rotten_feats = np.array([extract_features(img) for img in rotten_imgs])

    X = np.vstack([fresh_feats, rotten_feats])
    y = np.concatenate([np.ones(len(fresh_feats)), np.zeros(len(rotten_feats))])
    return X, y


def normalize(X: np.ndarray) -> tuple:
    """Min-Max normalise features to [0, 1]."""
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    return (X - x_min) / (x_max - x_min + 1e-8), x_min, x_max


# ── 3. ACTIVATION FUNCTIONS – HIDDEN LAYERS ───────────────────────────────────
def relu(z: np.ndarray) -> np.ndarray:
    """ReLU activation – used in all hidden layers during forward pass."""
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Sub-gradient of ReLU – used during back-propagation."""
    return (z > 0).astype(np.float32)


# ── 4. TRANSFER / ACTIVATION FUNCTIONS – OUTPUT LAYER ────────────────────────
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid (logistic) function – output in (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given its output a = sigmoid(z)."""
    return a * (1.0 - a)


def tanh_activation(z: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent – output in (-1, 1)."""
    return np.tanh(z)


def tanh_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative of tanh given its output a = tanh(z)."""
    return 1.0 - a ** 2


def hard_limit(z: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Hard-Limit (Heaviside step) transfer function.
    Returns 1 where z >= threshold, else 0.
    Used only for final binary decision – not used in training.
    """
    return (z >= threshold).astype(np.float32)


# ── 5. NEURAL NETWORK ─────────────────────────────────────────────────────────
class FruitNN:
    """
    Feedforward Neural Network: 18 → 16 → 8 → 4 → 1

    - Hidden layers use ReLU activation.
    - Output layer uses Sigmoid activation.
    - Trained with Binary Cross-Entropy loss and vanilla gradient descent.
    """

    def __init__(self, input_size: int = 18):
        # Layer dimensions: [input, h1=16, h2=8, h3=4, output=1]
        self.dims = [input_size, 16, 8, 4, 1]

        # He initialisation – optimal starting point for ReLU networks
        self.W = [
            np.random.randn(self.dims[i], self.dims[i + 1]).astype(np.float32)
            * np.sqrt(2.0 / self.dims[i])
            for i in range(len(self.dims) - 1)
        ]
        self.b = [
            np.zeros((1, self.dims[i + 1]), dtype=np.float32)
            for i in range(len(self.dims) - 1)
        ]

    # ── Forward Pass ──────────────────────────────────────────────────────────
    def forward(self, X: np.ndarray) -> tuple:
        """
        Forward pass through all layers.
        Returns (output, cache) where cache stores Z and A for back-prop.
        """
        cache = {"A0": X}

        # Hidden Layer 1 – 16 neurons – ReLU
        Z1 = X @ self.W[0] + self.b[0]
        A1 = relu(Z1)
        cache["Z1"], cache["A1"] = Z1, A1

        # Hidden Layer 2 – 8 neurons – ReLU
        Z2 = A1 @ self.W[1] + self.b[1]
        A2 = relu(Z2)
        cache["Z2"], cache["A2"] = Z2, A2

        # Hidden Layer 3 – 4 neurons – ReLU
        Z3 = A2 @ self.W[2] + self.b[2]
        A3 = relu(Z3)
        cache["Z3"], cache["A3"] = Z3, A3

        # Output Layer – 1 neuron – Sigmoid
        Z4 = A3 @ self.W[3] + self.b[3]
        A4 = sigmoid(Z4)
        cache["Z4"], cache["A4"] = Z4, A4

        return A4, cache

    # ── Loss ──────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        y_true = y_true.reshape(-1, 1)
        eps    = 1e-12
        return -np.mean(
            y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
        )

    # ── Backward Pass ─────────────────────────────────────────────────────────
    def backward(self, cache: dict, y_true: np.ndarray, lr: float) -> None:
        """Back-propagation: compute gradients and update weights in-place."""
        N      = cache["A0"].shape[0]
        y_true = y_true.reshape(-1, 1)

        # Output gradient (BCE + sigmoid derivative simplifies nicely)
        dZ4 = cache["A4"] - y_true

        dW3 = cache["A3"].T @ dZ4 / N
        db3 = dZ4.mean(axis=0, keepdims=True)
        dA3 = dZ4 @ self.W[3].T

        dZ3 = dA3 * relu_derivative(cache["Z3"])
        dW2 = cache["A2"].T @ dZ3 / N
        db2 = dZ3.mean(axis=0, keepdims=True)
        dA2 = dZ3 @ self.W[2].T

        dZ2 = dA2 * relu_derivative(cache["Z2"])
        dW1 = cache["A1"].T @ dZ2 / N
        db1 = dZ2.mean(axis=0, keepdims=True)
        dA1 = dZ2 @ self.W[1].T

        dZ1 = dA1 * relu_derivative(cache["Z1"])
        dW0 = cache["A0"].T @ dZ1 / N
        db0 = dZ1.mean(axis=0, keepdims=True)

        # Gradient descent weight update
        for i, (dW, db) in enumerate([(dW0, db0), (dW1, db1), (dW2, db2), (dW3, db3)]):
            self.W[i] -= lr * dW
            self.b[i] -= lr * db

    # ── Train ─────────────────────────────────────────────────────────────────
    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 200, lr: float = 0.01,
              print_every: int = 20) -> list:
        """Train for *epochs* epochs; return list of per-epoch losses."""
        loss_history = []
        for epoch in range(1, epochs + 1):
            y_pred, cache = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(cache, y, lr)
            loss_history.append(loss)
            if epoch % print_every == 0 or epoch == 1:
                print(f"  Epoch {epoch:>4d}/{epochs}  |  Loss: {loss:.6f}")
        return loss_history

    # ── Predict & Accuracy ────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return sigmoid probability for each sample. Shape: (N, 1)."""
        y_pred, _ = self.forward(X)
        return y_pred

    @staticmethod
    def accuracy(y_pred: np.ndarray, y_true: np.ndarray,
                 threshold: float = 0.5) -> float:
        """Fraction of correct binary predictions at *threshold*."""
        preds = (y_pred.flatten() >= threshold).astype(int)
        return np.mean(preds == y_true.astype(int))


# ── 6. TRANSFER FUNCTION DEMONSTRATION ────────────────────────────────────────
def demonstrate_transfer_functions(raw_logits: np.ndarray) -> None:
    """
    Apply and display Hard-Limit, Sigmoid, and Tanh on the raw logits (Z4).
    This shows the effect of each transfer function on the same pre-activation
    values before the final output decision is made.
    """
    print("\n" + "=" * 60)
    print("  TRANSFER / ACTIVATION FUNCTION DEMONSTRATIONS")
    print("=" * 60)

    sample = raw_logits[:10].flatten()
    print(f"\n  Raw logits (Z4, first 10 samples):")
    print(f"  {np.round(sample, 4)}")

    print(f"\n  1) Hard-Limit  (threshold = 0.0)  – binary step decision:")
    print(f"     {hard_limit(sample, threshold=0.0)}")

    print(f"\n  2) Sigmoid  – smooth probability in (0, 1):")
    print(f"     {np.round(sigmoid(sample), 4)}")

    print(f"\n  3) Tanh  – symmetric output in (-1, 1):")
    print(f"     {np.round(tanh_activation(sample), 4)}")

    print("=" * 60)


# ── 7. MAIN ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  FRUIT FRESHNESS NEURAL NETWORK  –  Lab 4")
    print("=" * 60)

    # Step 1 – Load images and extract features
    print("\n[1] Loading images & extracting features...")
    X_peach,       y_peach       = build_dataset("peach")
    X_strawberry,  y_strawberry  = build_dataset("strawberry")
    print(f"    Peach     : {len(X_peach)} samples  "
          f"(fresh={int(y_peach.sum())}, rotten={int((1-y_peach).sum())})")
    print(f"    Strawberry: {len(X_strawberry)} samples  "
          f"(fresh={int(y_strawberry.sum())}, rotten={int((1-y_strawberry).sum())})")

    # Step 2 – Align sample counts, concatenate both feature vectors
    n     = min(len(X_peach), len(X_strawberry))
    idx_p = np.random.permutation(len(X_peach))[:n]
    idx_s = np.random.permutation(len(X_strawberry))[:n]

    # Input 1 (9 features) ++ Input 2 (9 features) = 18-dim input vector
    X_combined = np.hstack([X_peach[idx_p], X_strawberry[idx_s]])
    # Label: 1 only when both fruits are fresh (AND logic)
    y_combined = y_peach[idx_p] * y_strawberry[idx_s]

    print(f"\n    Combined input : {X_combined.shape}  (N x 18 features)")
    print(f"    Label balance  : both-fresh={int(y_combined.sum())}, "
          f"otherwise={int((1-y_combined).sum())}")

    # Step 3 – Normalise features to [0, 1]
    X_norm, _, _ = normalize(X_combined)

    # Step 4 – 80 / 20 train-test split
    perm  = np.random.permutation(n)
    split = int(0.8 * n)
    X_train, y_train = X_norm[perm[:split]], y_combined[perm[:split]]
    X_test,  y_test  = X_norm[perm[split:]], y_combined[perm[split:]]
    print(f"\n    Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

    # Step 5 – Build and train the network
    print("\n[2] Network Architecture  :  18 --> 16 --> 8 --> 4 --> 1")
    print("    Hidden Activations     :  ReLU")
    print("    Output Activation      :  Sigmoid")
    print("    Loss Function          :  Binary Cross-Entropy")
    print("    Optimiser              :  Gradient Descent  (lr=0.01)\n")

    nn = FruitNN(input_size=18)

    print("[3] Training (200 epochs)...\n")
    nn.train(X_train, y_train, epochs=200, lr=0.01, print_every=20)

    # Step 6 – Evaluate on test set
    print("\n[4] Evaluation on Test Set")
    y_pred = nn.predict(X_test)
    acc    = nn.accuracy(y_pred, y_test)
    loss   = nn.compute_loss(y_pred, y_test)
    print(f"    Test Loss     : {loss:.6f}")
    print(f"    Test Accuracy : {acc * 100:.2f}%")

    # Step 7 – Demonstrate transfer / activation functions
    print("\n[5] Transfer & Activation Function Demonstrations")
    _, cache = nn.forward(X_test)
    demonstrate_transfer_functions(cache["Z4"])

    # Step 8 – Per-sample prediction table
    print("\n[6] Sample Predictions (first 10 test samples)")
    print(f"  {'#':<4} {'Sigmoid':>10} {'Hard-Limit':>12} {'Tanh':>10} {'True Label':>12}")
    print("  " + "-" * 52)
    for i in range(min(10, len(y_test))):
        z   = float(cache["Z4"][i, 0])
        sig = float(sigmoid(np.array(z)))
        hl  = int(hard_limit(np.array(z), threshold=0.0))
        th  = float(tanh_activation(np.array(z)))
        print(f"  {i+1:<4} {sig:>10.4f} {hl:>12} {th:>10.4f} {int(y_test[i]):>12}")

    print("\n[Done] Neural Network training and evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
