import edu1sces.core
from edu1sces.model.types import Site


class SolverResult:
    """Result of solving a model.

    Wraps the core SolverResult and provides convenient methods
    for computing expectation values.

    Attributes:
        energies: Eigenvalues (energies) for each computed state.
        eigenvectors: Eigenvectors for each computed state.
        lanczos_logs: Lanczos solver logs for each state.
        inverse_iteration_logs: Inverse iteration solver logs for each state.
    """

    def __init__(
        self,
        core_result: edu1sces.core.SolverResult,
        site_to_integer: dict[Site, int],
        default_num_threads: int,
    ) -> None:
        self._core_result = core_result
        self._site_to_integer = site_to_integer
        self._default_num_threads = default_num_threads

    @property
    def energies(self) -> list[float]:
        """Eigenvalues (energies) for each computed state."""
        return self._core_result.energies

    @property
    def eigenvectors(self) -> list[list[float]]:
        """Eigenvectors for each computed state."""
        return self._core_result.eigenvectors

    @property
    def lanczos_logs(self) -> list[edu1sces.core.LanczosLog]:
        """Lanczos solver logs for each state."""
        return self._core_result.lanczos_logs

    @property
    def inverse_iteration_logs(self) -> list[edu1sces.core.InverseIterationLog]:
        """Inverse iteration solver logs for each state."""
        return self._core_result.inverse_iteration_logs

    @property
    def num_states(self) -> int:
        """Number of computed eigenstates."""
        return len(self._core_result.energies)

    def expectation_onsite(
        self,
        local_op: edu1sces.core.CsrMatrix,
        site: Site,
        state_index: int,
        *,
        num_threads: int | None = None,
    ) -> float:
        """Compute expectation value of a local operator at a specific site.

        Args:
            local_op: CSR matrix representing the local operator.
            site: Site for which to compute the expectation value.
            num_threads: Number of threads for parallel computation.
                If None, uses the value from SolverParameters.
            state_index: Index of the eigenstate (0 = ground state).

        Returns:
            Expectation value <psi| O_site |psi>.
        """
        site_index = self._site_to_integer[site]
        threads = num_threads if num_threads is not None else self._default_num_threads
        return self._core_result.expectation_onsite(
            local_op, site_index, threads, state_index
        )

    def correlation_function(
        self,
        op1: edu1sces.core.CsrMatrix,
        site1: Site,
        op2: edu1sces.core.CsrMatrix,
        site2: Site,
        state_index: int,
        *,
        num_threads: int | None = None,
    ) -> float:
        """Compute two-point correlation function <psi|O1_{site1} O2_{site2}|psi>.

        When the operators change quantum numbers (e.g., S+, S-), the intermediate
        basis is automatically constructed and cached.

        Args:
            op1: CSR matrix representing local operator 1.
            site1: Site index for operator 1.
            op2: CSR matrix representing local operator 2.
            site2: Site index for operator 2.
            num_threads: Number of threads for parallel computation.
                If None, uses the value from SolverParameters.
            state_index: Index of the eigenstate (0 = ground state).

        Returns:
            Correlation value <psi|O1_{site1} O2_{site2}|psi>.
        """
        site1_index = self._site_to_integer[site1]
        site2_index = self._site_to_integer[site2]
        threads = num_threads if num_threads is not None else self._default_num_threads
        return self._core_result.correlation_function(
            op1, site1_index, op2, site2_index, threads, state_index
        )
