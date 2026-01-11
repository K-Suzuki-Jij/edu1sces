import edu1sces.core


class ConjugateGradientParameters:
    """Parameters for the conjugate gradient solver.

    Args:
        residual_tol: Convergence threshold for residual norm.
        max_step: Maximum number of CG iterations.
    """

    def __init__(
        self,
        residual_tol: float = 1e-12,
        max_step: int = 1000,
    ) -> None:
        if residual_tol <= 0:
            raise ValueError("residual_tol must be positive")
        if max_step <= 0:
            raise ValueError("max_step must be positive")

        self.residual_tol = residual_tol
        self.max_step = max_step

        self.core_params = edu1sces.core.ConjugateGradientParameters(
            residual_tol,
            max_step,
        )


class InverseIterationParameters:
    """Parameters for the inverse iteration solver.

    Args:
        diag_add: Diagonal shift to ensure positive definiteness.
        eigenvec_tol: Convergence threshold for eigenvector residual.
        max_step: Maximum number of inverse iteration steps.
        cg_params: Parameters for the inner CG solver.
    """

    def __init__(
        self,
        diag_add: float = 1e-7,
        eigenvec_tol: float = 1e-8,
        max_step: int = 100,
        cg_params: ConjugateGradientParameters | None = None,
    ) -> None:
        if diag_add <= 0:
            raise ValueError("diag_add must be positive")
        if eigenvec_tol <= 0:
            raise ValueError("eigenvec_tol must be positive")
        if max_step <= 0:
            raise ValueError("max_step must be positive")

        if cg_params is None:
            cg_params = ConjugateGradientParameters()

        self.diag_add = diag_add
        self.eigenvec_tol = eigenvec_tol
        self.max_step = max_step
        self.cg_params = cg_params

        self.core_params = edu1sces.core.InverseIterationParameters(
            diag_add,
            eigenvec_tol,
            max_step,
            cg_params.core_params,
        )


class SolverParameters:
    """Parameters for the eigenvalue solver.

    Args:
        eigenvalue_tol: Convergence threshold for eigenvalue (Lanczos).
        min_step: Minimum Lanczos iterations.
        max_step: Maximum Lanczos iterations.
        num_threads: Number of threads for parallel computation.
        inverse_iteration_params: Parameters for eigenvector refinement.
    """

    def __init__(
        self,
        eigenvalue_tol: float = 1e-14,
        min_step: int = 5,
        max_step: int = 1000,
        num_threads: int = 1,
        inverse_iteration_params: InverseIterationParameters | None = None,
    ) -> None:
        if eigenvalue_tol <= 0:
            raise ValueError("eigenvalue_tol must be positive")
        if min_step <= 0:
            raise ValueError("min_step must be positive")
        if max_step <= 0:
            raise ValueError("max_step must be positive")
        if max_step < min_step:
            raise ValueError("max_step must be >= min_step")
        if num_threads <= 0:
            raise ValueError("num_threads must be positive")

        if inverse_iteration_params is None:
            inverse_iteration_params = InverseIterationParameters()

        self.eigenvalue_tol = eigenvalue_tol
        self.min_step = min_step
        self.max_step = max_step
        self.num_threads = num_threads
        self.inverse_iteration_params = inverse_iteration_params

        self.core_params = edu1sces.core.SolverParameters(
            eigenvalue_tol,
            min_step,
            max_step,
            num_threads,
            inverse_iteration_params.core_params,
        )
