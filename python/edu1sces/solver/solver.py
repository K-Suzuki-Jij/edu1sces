import edu1sces.core
from edu1sces.model import HeisenbergModel, HubbardModel, KondoLatticeModel
from .solver_parameters import SolverParameters
from .solver_result import SolverResult


def solve(
    model: HeisenbergModel | HubbardModel | KondoLatticeModel,
    *,
    total_sz: float,
    num_electrons: int | None = None,
    params: SolverParameters | None = None,
) -> SolverResult:
    """Solve the model to find the ground state.

    Args:
        model: HeisenbergModel, HubbardModel, or KondoLatticeModel.
        total_sz: Target total Sz quantum number.
        num_electrons: Number of electrons (required for HubbardModel and KondoLatticeModel).
        params: Solver parameters (uses defaults if None).

    Returns:
        SolverResult with energy, eigenvector, and logs.
    """
    if params is None:
        params = SolverParameters()

    if isinstance(model, HeisenbergModel):
        core_result = edu1sces.core.solve_heisenberg(
            model.core_model,
            total_sz,
            params.core_params,
        )
        return SolverResult(
            core_result,
            model.site_to_integer,
            params.num_threads,
        )
    elif isinstance(model, HubbardModel):
        if num_electrons is None:
            raise ValueError("num_electrons is required for HubbardModel")
        core_result = edu1sces.core.solve_hubbard(
            model.core_model,
            num_electrons,
            total_sz,
            params.core_params,
        )
        return SolverResult(
            core_result,
            model.site_to_integer,
            params.num_threads,
        )
    elif isinstance(model, KondoLatticeModel):
        if num_electrons is None:
            raise ValueError("num_electrons is required for KondoLatticeModel")
        core_result = edu1sces.core.solve_kondo_lattice(
            model.core_model,
            num_electrons,
            total_sz,
            params.core_params,
        )
        return SolverResult(
            core_result,
            model.site_to_integer,
            params.num_threads,
        )
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")
