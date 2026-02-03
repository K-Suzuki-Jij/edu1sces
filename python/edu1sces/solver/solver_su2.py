import edu1sces.core
from edu1sces.model import SU2HeisenbergModel
from .solver_parameters import SolverParameters
from .solver_result import SU2SolverResult


def solve_su2(
    model: SU2HeisenbergModel,
    *,
    total_s: float,
    params: SolverParameters | None = None,
) -> SU2SolverResult:
    """Solve an SU(2) symmetric Heisenberg model to find the ground state.

    Args:
        model: SU2HeisenbergModel.
        total_s: Target total spin quantum number S.
        params: Solver parameters (uses defaults if None).

    Returns:
        SU2SolverResult with energy, eigenvector, and logs.
    """
    if params is None:
        params = SolverParameters(num_states=1)

    core_result = edu1sces.core.solve_su2_heisenberg(
        model.core_model,
        total_s,
        params.core_params,
    )
    return SU2SolverResult(
        core_result,
        model.site_to_integer,
        params.num_threads,
    )
