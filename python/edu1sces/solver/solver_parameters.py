from dataclasses import dataclass

import edu1sces.core


@dataclass
class ConjugateGradientParameters:
    """Parameters for the conjugate gradient solver.

    Args:
        residual_tol: Convergence threshold for residual norm.
        max_step: Maximum number of CG iterations.
    """

    residual_tol: float = 1e-12
    max_step: int = 1000

    def __post_init__(self) -> None:
        if self.residual_tol <= 0:
            raise ValueError("residual_tol must be positive")
        if self.max_step <= 0:
            raise ValueError("max_step must be positive")


@dataclass
class InverseIterationParameters:
    """Parameters for the inverse iteration solver.

    Args:
        diag_add: Diagonal shift to ensure positive definiteness.
        eigenvec_tol: Convergence threshold for eigenvector residual.
        max_step: Maximum number of inverse iteration steps.
        cg_params: Parameters for the inner CG solver.
    """

    diag_add: float = 1e-7
    eigenvec_tol: float = 1e-8
    max_step: int = 100
    cg_params: ConjugateGradientParameters | None = None

    def __post_init__(self) -> None:
        if self.diag_add <= 0:
            raise ValueError("diag_add must be positive")
        if self.eigenvec_tol <= 0:
            raise ValueError("eigenvec_tol must be positive")
        if self.max_step <= 0:
            raise ValueError("max_step must be positive")

        if self.cg_params is None:
            self.cg_params = ConjugateGradientParameters()


@dataclass
class SolverParameters:
    """Parameters for the eigenvalue solver.

    Args:
        num_states: Number of eigenstates to compute (1 = ground state only).
        eigenvalue_tol: Convergence threshold for eigenvalue (Lanczos).
        min_step: Minimum Lanczos iterations.
        max_step: Maximum Lanczos iterations.
        num_threads: Number of threads for parallel computation.
        inverse_iteration_params: Parameters for eigenvector refinement.
        output_log: If True, print progress to stdout with real-time updates.
    """

    num_states: int
    eigenvalue_tol: float = 1e-14
    min_step: int = 5
    max_step: int = 1000
    num_threads: int = 1
    inverse_iteration_params: InverseIterationParameters | None = None
    output_log: bool = False

    def __post_init__(self) -> None:
        if self.eigenvalue_tol <= 0:
            raise ValueError("eigenvalue_tol must be positive")
        if self.min_step <= 0:
            raise ValueError("min_step must be positive")
        if self.max_step <= 0:
            raise ValueError("max_step must be positive")
        if self.max_step < self.min_step:
            raise ValueError("max_step must be >= min_step")
        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive")
        if self.num_states <= 0:
            raise ValueError("num_states must be positive")

        if self.inverse_iteration_params is None:
            self.inverse_iteration_params = InverseIterationParameters()

        # Build core params with output_log propagated to CG
        cg = self.inverse_iteration_params.cg_params
        cg_core = edu1sces.core.ConjugateGradientParameters(
            cg.residual_tol,
            cg.max_step,
            self.output_log,
        )
        inv = self.inverse_iteration_params
        inv_core = edu1sces.core.InverseIterationParameters(
            inv.diag_add,
            inv.eigenvec_tol,
            inv.max_step,
            cg_core,
        )
        self.core_params = edu1sces.core.SolverParameters(
            self.eigenvalue_tol,
            self.min_step,
            self.max_step,
            self.num_threads,
            inv_core,
            self.output_log,
            self.num_states,
        )
