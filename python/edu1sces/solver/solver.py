import edu1sces.core
from edu1sces.model import HeisenbergModel, HubbardModel
from .solver_parameters import SolverParameters


def solve(
    model: HeisenbergModel | HubbardModel,
    *,
    total_sz: float,
    num_electrons: int | None = None,
    params: SolverParameters | None = None,
) -> edu1sces.core.SolverResult:
    """Solve the model to find the ground state.

    Args:
        model: HeisenbergModel or HubbardModel.
        total_sz: Target total Sz quantum number.
        num_electrons: Number of electrons (required for HubbardModel).
        params: Solver parameters (uses defaults if None).

    Returns:
        SolverResult with energy, eigenvector, and logs.
    """
    if params is None:
        params = SolverParameters()

    if isinstance(model, HeisenbergModel):
        return edu1sces.core.solve_heisenberg(
            model.core_model,
            total_sz,
            params.core_params,
        )
    elif isinstance(model, HubbardModel):
        if num_electrons is None:
            raise ValueError("num_electrons is required for HubbardModel")
        return edu1sces.core.solve_hubbard(
            model.core_model,
            num_electrons,
            total_sz,
            params.core_params,
        )
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")
