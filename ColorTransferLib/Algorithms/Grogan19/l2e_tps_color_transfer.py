import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.optim import Adam


# ============================================================
# NumPy <-> Torch helper functions
# ============================================================

def img_np_to_tensor(img_np, device):
    """
    Converts a NumPy image (H, W, 3) to a flat Torch tensor (N, 3) in float32 [0, 1].
    """
    if img_np.dtype == np.uint8:
        arr = img_np.astype(np.float32) / 255.0
    else:
        arr = np.clip(img_np.astype(np.float32), 0.0, 1.0)

    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)
    flat_t = torch.from_numpy(flat).to(device)
    return flat_t, (h, w)


def tensor_to_np_img(t, hw):
    """
    Converts a flat Torch tensor (N, 3) in float32 [0, 1] to a NumPy image (H, W, 3) uint8.
    """
    h, w = hw
    arr = t.clamp(0.0, 1.0).detach().cpu().numpy().reshape(h, w, 3)
    return (arr * 255.0 + 0.5).astype(np.uint8)


# ============================================================
# TPS (Paper): ψ(r) = -r   (Eq. 9)
# ============================================================

def tps_basis(r: torch.Tensor) -> torch.Tensor:
    """
    Thin-plate spline RBF used in the paper (Eq. 9):

        ψ(||x - c_j||) = -||x - c_j||
    """
    return -r


class TPSWarp(nn.Module):
    """
    φ(x) = A x + o + Σ_j w_j ψ(||x - c_j||),

    where:
        A ∈ R^{3x3}, o ∈ R^3,
        W ∈ R^{m x 3}, C ∈ R^{m x 3}.
    """

    def __init__(self, control_points: torch.Tensor):
        super().__init__()
        self.register_buffer("C", control_points)  # (m,3)
        m = control_points.shape[0]

        self.A = nn.Parameter(torch.eye(3, dtype=torch.float32))
        self.o = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.W = nn.Parameter(torch.zeros(m, 3, dtype=torch.float32))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor
            Shape (N, 3), RGB in [0, 1].

        Returns
        -------
        torch.Tensor
            Warped colours φ(X) with shape (N, 3).
        """
        Y = X @ self.A.T + self.o  # affine Teil

        diff = X[:, None, :] - self.C[None, :, :]   # (N,m,3)
        r = torch.linalg.norm(diff, dim=-1)         # (N,m)
        psi = tps_basis(r)                          # (N,m)
        Y = Y + psi @ self.W                        # (N,3)

        return Y


# ============================================================
# L2E colour transfer (paper-like, with adaptive h)
# ============================================================

class L2EColorTransferPaper:
    """
    Paper-like implementation of Grogan & Dahyot (2019):

      - GMMs with isotropic covariances Σ = h^2 I (Eq. 8)
      - TPS transfer function φ_θ(x) with ψ(r) = -r (Eq. 9)
      - L2E cost according to Eq. (10) and (11) (up to constant factors)
      - Single regularisation term: λ * ||W||^2 (roughness penalty, cf. Eq. (13))
      - Bandwidth h is:
          * initially estimated from K-means centres
          * updated via an annealing scheme (decreasing h)
    """

    def __init__(
        self,
        K: int = 50,
        grid_size: int = 5,
        lambda_w: float = 1e-3,
        bandwidth_alpha: float = 0.3,
        anneal_factor: float = 0.5,
        anneal_stages: int = 3,
        device: str | None = None,
    ):
        """
        Parameters
        ----------
        K : int
            Number of GMM components (K-means centres) for target and palette.
        grid_size : int
            TPS control points on a regular grid in the RGB cube [0, 1]^3 (m = grid_size^3).
        lambda_w : float
            Weight for roughness penalty ||W||^2 (proxy for ∫||D^2 φ||^2 dx, cf. Eq. (13)).
        bandwidth_alpha : float
            Factor α for initial bandwidth estimation:
                h0 = α * sqrt( median(||μ_ti - μ_tj||^2) ).
        anneal_factor : float
            Factor < 1.0 for stepwise reduction of h (simulated annealing).
        anneal_stages : int
            Number of h stages (outer loop).
        device : {"cpu", "cuda"} or None
            Device to run Torch on, default chooses CUDA if available.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.K = K
        self.grid_size = grid_size
        self.lambda_w = lambda_w
        self.bandwidth_alpha = bandwidth_alpha
        self.anneal_factor = anneal_factor
        self.anneal_stages = anneal_stages

        self.tps: TPSWarp | None = None
        self.mu_t: torch.Tensor | None = None
        self.mu_p: torch.Tensor | None = None

        self.h: float = 0.15  # wird nachher überschrieben
        self.fitted: bool = False

    # --------------------------------------------------------
    # Control-point grid (m = grid_size^3) in RGB cube [0, 1]^3
    # --------------------------------------------------------
    def _create_grid(self) -> torch.Tensor:
        g = torch.linspace(0.0, 1.0, self.grid_size, device=self.device)
        grid = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), dim=-1)
        return grid.reshape(-1, 3)  # (m,3)

    # --------------------------------------------------------
    # Estimate bandwidth h from μ_t (L2E-style heuristic)
    # --------------------------------------------------------
    def _estimate_initial_bandwidth(self, mu_t: torch.Tensor) -> float:
        """
        h0 = α * sqrt( median(||μ_ti - μ_tj||^2) ), i != j

        Standard bandwidth heuristic for kernel densities and L2E matching.
        """
        with torch.no_grad():
            diff = mu_t[:, None, :] - mu_t[None, :, :]
            d2 = (diff * diff).sum(dim=-1)
            # nur i != j
            K_ = d2.shape[0]
            if K_ > 1:
                d2_vec = d2[~torch.eye(K_, dtype=bool, device=d2.device)]
                med = torch.median(d2_vec)
            else:
                med = torch.tensor(1e-2, device=mu_t.device)

            h0 = torch.sqrt(med.clamp(min=1e-8)) * self.bandwidth_alpha
            return float(h0.item())

    # --------------------------------------------------------
    # L2E cost (Eq. 10–11) + λ||W||^2
    # --------------------------------------------------------
    def _l2e_cost(self) -> torch.Tensor:
        """
        C(θ) ≈ ‖p_t‖² - 2 ⟨p_t | p_p⟩ + λ ||W||²

        Approximations:
          ‖p_t‖²  ≈ mean_{i,k} exp(-||φ(μ_ti) - φ(μ_tk)||² / (4 h²))
          ⟨p_t|p_p⟩ ≈ mean_{i,k} exp(-||φ(μ_tk) - μ_pi||² / (4 h²))

        The 1/K² constants are omitted; they do not change the minimiser.
        """
        h2 = self.h * self.h

        mu_t = self.mu_t          # (K,3)
        mu_p = self.mu_p          # (K,3)
        phi_t = self.tps(mu_t)    # (K,3)

        # Term ‖p_t‖²
        diff_tt = phi_t[:, None, :] - phi_t[None, :, :]  # (K,K,3)
        d2_tt = (diff_tt * diff_tt).sum(dim=-1)          # (K,K)
        k_tt = torch.exp(-d2_tt / (4.0 * h2))
        term1 = k_tt.mean()

        # Term ⟨p_t|p_p⟩
        diff_tp = phi_t[:, None, :] - mu_p[None, :, :]   # (K,K,3)
        d2_tp = (diff_tp * diff_tp).sum(dim=-1)          # (K,K)
        k_tp = torch.exp(-d2_tp / (4.0 * h2))
        term2 = k_tp.mean()

        # Roughness-Penalty (Proxy für ∫||D² φ||² dx, Eq. (13))
        reg_w = (self.tps.W * self.tps.W).sum()

        return term1 - 2.0 * term2 + self.lambda_w * reg_w

    # --------------------------------------------------------
    # Optimisation with annealing over h
    # --------------------------------------------------------
    def _optimize(self, total_iters: int, lr: float, verbose: bool):
        """
        Simulated annealing over the bandwidth h (Section 3.3):

        - Start at h0
        - Decrease h in stages: h_s = h0 * (anneal_factor^s)
        - Run a few GD/Adam steps on C(θ) per stage
        """
        if self.anneal_stages < 1:
            self.anneal_stages = 1

        iters_per_stage = max(1, total_iters // self.anneal_stages)
        h0 = self.h
        opt = Adam(self.tps.parameters(), lr=lr)

        for stage in range(self.anneal_stages):
            self.h = h0 * (self.anneal_factor ** stage)

            for it in range(iters_per_stage):
                opt.zero_grad()
                cost = self._l2e_cost()
                cost.backward()
                opt.step()

                if verbose and (it % 1000 == 0 or it == iters_per_stage - 1):
                    global_it = stage * iters_per_stage + it
                    print(
                        f"[L2E] stage {stage+1}/{self.anneal_stages}, "
                        f"iter {global_it:5d}, h={self.h:.5f}, cost={cost.item():.6f}"
                    )

    # --------------------------------------------------------
    # Training (K-means + bandwidth estimation + L2E optimisation)
    # --------------------------------------------------------
    def fit_from_numpy(
        self,
        target_np: np.ndarray,
        palette_np: np.ndarray,
        n_iters: int = 200,
        lr: float = 0.05,
        verbose: bool = True,
    ):
        """
        Estimate θ = (A, o, W) from target and palette images (NumPy, RGB).

        Parameters
        ----------
        target_np : np.ndarray
            Target image, shape (H, W, 3), uint8 or float in [0, 1].
        palette_np : np.ndarray
            Palette image, shape (H, W, 3), uint8 or float in [0, 1].
        """
        # 1) images → flat tensors
        T_flat, _ = img_np_to_tensor(target_np, self.device)
        P_flat, _ = img_np_to_tensor(palette_np, self.device)

        # 2) K-means in RGB (GMM means, cf. Sec. 3.1)
        kmeans_t = KMeans(n_clusters=self.K, n_init=5, random_state=0).fit(
            T_flat.cpu().numpy()
        )
        kmeans_p = KMeans(n_clusters=self.K, n_init=5, random_state=0).fit(
            P_flat.cpu().numpy()
        )

        self.mu_t = torch.tensor(
            kmeans_t.cluster_centers_, dtype=torch.float32, device=self.device
        )
        self.mu_p = torch.tensor(
            kmeans_p.cluster_centers_, dtype=torch.float32, device=self.device
        )

        # 3) initiale Bandbreite h0 schätzen (aus μ_t)
        self.h = self._estimate_initial_bandwidth(self.mu_t)

        if verbose:
            print(f"[L2E] initial bandwidth h0 = {self.h:.5f}")

        # 4) TPS-Kontrollpunkte auf regulärem RGB-Gitter (m = grid_size^3)
        C = self._create_grid()
        self.tps = TPSWarp(C).to(self.device)

        # 5) L2E-Optimierung mit Annealing über h
        self._optimize(total_iters=n_iters, lr=lr, verbose=verbose)

        self.fitted = True
        return self

    # --------------------------------------------------------
    # Application: φ_θ on target image
    # --------------------------------------------------------
    def transfer_numpy(self, target_np: np.ndarray) -> np.ndarray:
        """
        Apply the learned TPS warp to a target image.

        Returns
        -------
        np.ndarray
            Warped image, shape (H, W, 3), dtype uint8.
        """
        assert self.fitted, "Must call fit_from_numpy() before transfer_numpy()."

        T_flat, hw = img_np_to_tensor(target_np, self.device)
        with torch.no_grad():
            recol_flat = self.tps(T_flat)

        return tensor_to_np_img(recol_flat, hw)


# ------------------------------------------------------------------------------------------------------------------
# Minimal self-test
# ------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    H, W = 64, 64
    target_np = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    palette_np = (np.random.rand(H, W, 3) * 255).astype(np.uint8)

    model = L2EColorTransferPaper(
        K=50,
        grid_size=5,
        lambda_w=1e-3,
        bandwidth_alpha=0.3,
        anneal_factor=0.5,
        anneal_stages=3,
        device=None,
    )

    model.fit_from_numpy(target_np, palette_np, n_iters=180, lr=0.05, verbose=True)
    out_np = model.transfer_numpy(target_np)
    print("Result:", out_np.shape, out_np.dtype)
