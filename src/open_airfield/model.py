"""U4 — reconstruction model v0 (technical spec §U4).

Physics-constrained coordinate network: FullyConnected (x,y,z) -> (u,v,w,p),
p auxiliary and never scored. Loss = w_d*data + w_c*continuity^2 + w_b*bc.

- PDE: own subclass of physicsnemo.sym.eq.pde.PDE (the container ships no
  library PDE module — verified 24 Aug), written in sympy following the
  ldc_pinns pattern at the container's commit (7e08f49). Momentum residuals
  exist in the class but are loss-gated OFF by default (design invariant 3):
  molecular nu on a RANS mean field mis-states the turbulent stresses.
- Residuals: PhysicsInformer(required_outputs=[...], equations=pde,
  grad_method="autodiff") — 3D is native, dim comes from the equations object.
- Collocation: 4,096 torch.rand points per step scaled to the box (no mesh).
- BC per truth source: synthetic -> match truth boundary values at sampled
  wall points (data-consistent); CASE-01 -> no-slip walls + declared supply
  velocity (built by the caller, passed as bc_points/bc_values).
- Training envelope: Adam + exp decay (the ldc constant), <=20k steps, NaN
  guard trips the run red.

Runs inside the physicsnemo:26.06 container on the T4; this module is not
importable on the Mac (no torch/physicsnemo locally) and its tests skip there.
"""

import math
import time

import numpy as np
import torch
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from sympy import Function, Number, Symbol

from open_airfield.contracts import ObservationSet
from open_airfield.geometry import LX, LY, LZ


class SteadyIncompressible3D(PDE):
    """3D steady incompressible Navier-Stokes, sympy, ldc pattern.

    Continuity is the v1 residual; momentum is carried but off by default.
    """

    def __init__(self, nu: float = 1.5e-5, rho: float = 1.0):
        self.dim = 3  # PhysicsInformer reads equations.dim (phy_informer.py:122)
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        iv = {"x": x, "y": y, "z": z}
        u = Function("u")(*iv.values())
        v = Function("v")(*iv.values())
        w = Function("w")(*iv.values())
        p = Function("p")(*iv.values())
        nu, rho = Number(nu), Number(rho)
        adv = lambda f: u * f.diff(x) + v * f.diff(y) + w * f.diff(z)  # noqa: E731
        lap = lambda f: f.diff(x, 2) + f.diff(y, 2) + f.diff(z, 2)  # noqa: E731
        self.equations = {
            "continuity": u.diff(x) + v.diff(y) + w.diff(z),
            "momentum_x": adv(u) + (1 / rho) * p.diff(x) - nu * lap(u),
            "momentum_y": adv(v) + (1 / rho) * p.diff(y) - nu * lap(v),
            "momentum_z": adv(w) + (1 / rho) * p.diff(z) - nu * lap(w),
        }


class TrainingDiverged(RuntimeError):
    """NaN guard: trips the run red instead of logging garbage."""


class PINNReconstructor:
    """Reconstructor: fit() trains the network, predict() evaluates it."""

    def __init__(
        self,
        bc_points: np.ndarray,
        bc_values: np.ndarray,
        steps: int = 20_000,
        collocation: int = 4_096,
        lr: float = 1e-3,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),  # (data, continuity, bc)
        momentum: bool = False,  # invariant 3: experiment flag, OFF by default
        nu: float = 1.5e-5,
        seed: int = 0,
        log_every: int = 500,
        device: str | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.steps = steps
        self.collocation = collocation
        self.lr = lr
        self.w_data, self.w_cont, self.w_bc = weights
        self.momentum = momentum
        self.seed = seed
        self.log_every = log_every
        self.log: list[dict] = []

        torch.manual_seed(seed)
        self.model = FullyConnected(
            in_features=3, out_features=4, num_layers=6, layer_size=512
        ).to(self.device)

        pde = SteadyIncompressible3D(nu=nu)
        required = ["continuity"] + (
            ["momentum_x", "momentum_y", "momentum_z"] if momentum else []
        )
        self.informer = PhysicsInformer(
            required_outputs=required,
            equations=pde,
            grad_method="autodiff",
            device=self.device,
        )

        self._bc_pts = torch.tensor(bc_points, dtype=torch.float32, device=self.device)
        self._bc_u = torch.tensor(bc_values, dtype=torch.float32, device=self.device)
        self._box = torch.tensor([LX, LY, LZ], device=self.device)

    def _residuals(self, coords: torch.Tensor) -> dict:
        out = self.model(coords)
        return self.informer.forward(
            {
                "coordinates": coords,
                "u": out[:, 0:1],
                "v": out[:, 1:2],
                "w": out[:, 2:3],
                "p": out[:, 3:4],
            }
        )

    def fit(self, obs: ObservationSet) -> None:
        obs_pts = torch.tensor(obs.points, dtype=torch.float32, device=self.device)
        obs_u = torch.tensor(obs.u, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 0.9999871767586216**step  # ldc constant
        )

        t0 = time.perf_counter()
        for step in range(self.steps):
            optimizer.zero_grad()

            colloc = torch.rand(
                self.collocation, 3, device=self.device, requires_grad=True
            ) * self._box
            residuals = self._residuals(colloc)
            cont_loss = torch.mean(residuals["continuity"] ** 2)
            mom_loss = (
                sum(
                    torch.mean(residuals[k] ** 2)
                    for k in ("momentum_x", "momentum_y", "momentum_z")
                )
                if self.momentum
                else torch.tensor(0.0, device=self.device)
            )

            data_loss = torch.mean((self.model(obs_pts)[:, :3] - obs_u) ** 2)
            bc_loss = torch.mean((self.model(self._bc_pts)[:, :3] - self._bc_u) ** 2)

            loss = (
                self.w_data * data_loss
                + self.w_cont * (cont_loss + mom_loss)
                + self.w_bc * bc_loss
            )
            if not math.isfinite(loss.item()):
                raise TrainingDiverged(f"non-finite loss at step {step}")

            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % self.log_every == 0 or step == self.steps - 1:
                row = {
                    "step": step,
                    "loss": loss.item(),
                    "data": data_loss.item(),
                    "continuity": cont_loss.item(),
                    "bc": bc_loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": round(time.perf_counter() - t0, 1),
                }
                self.log.append(row)
                print(
                    f"step {row['step']:>6} loss {row['loss']:.3e} "
                    f"(data {row['data']:.3e} cont {row['continuity']:.3e} "
                    f"bc {row['bc']:.3e}) {row['elapsed_s']}s",
                    flush=True,
                )

    @torch.no_grad()
    def predict(self, points: np.ndarray, batch: int = 65_536) -> np.ndarray:
        self.model.eval()
        chunks = []
        for i in range(0, len(points), batch):
            pts = torch.tensor(
                points[i : i + batch], dtype=torch.float32, device=self.device
            )
            chunks.append(self.model(pts)[:, :3].cpu().numpy())
        self.model.train()
        return np.vstack(chunks).astype(np.float64)
