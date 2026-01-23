import edu1sces.core
from .utils import (
    build_site_index,
    convert_onsite_potential_to_array,
    convert_intersite_interactions_to_indexed_dict,
)
from .types import Site, Bond


class SU2HeisenbergModel:
    """SU(2) symmetric Heisenberg model.

    Only isotropic exchange interactions J * S_i · S_j are allowed.
    No Zeeman field or single-ion anisotropy (they break SU(2) symmetry).

    The Hamiltonian is:
        H = Σ_{i<j} J_{ij} S_i · S_j

    Attributes:
        site_to_integer: Mapping from user-defined site labels to integer indices.
        num_sites: Number of lattice sites.
        core_model: The underlying Rust core model.
    """

    def __init__(
        self,
        spins: dict[Site, float],
        exchange: dict[Bond, float] | None = None,
    ) -> None:
        """Initialize an SU(2) symmetric Heisenberg model.

        Args:
            spins: Dictionary mapping sites to spin quantum numbers (e.g., 0.5 for spin-1/2).
            exchange: Isotropic exchange coupling J_{ij} for S_i · S_j interactions.
        """
        if exchange is None:
            exchange = {}

        self.site_to_integer = build_site_index(spins.keys())
        self.num_sites = len(self.site_to_integer)
        if self.num_sites == 0:
            raise ValueError("site_list must be non-empty")

        self.spin_list = convert_onsite_potential_to_array(self.site_to_integer, spins)

        self.exchange = convert_intersite_interactions_to_indexed_dict(
            self.site_to_integer, exchange, directed=False
        )

        self.core_model = edu1sces.core.SU2HeisenbergModel(
            self.spin_list,
            self.exchange,
        )

    def calc_dim_su2_sector(self, total_s: float) -> int:
        """Calculate the dimension of the SU(2) sector with the given total spin.

        Args:
            total_s: Total spin value of the sector (integer or half-integer).

        Returns:
            Dimension of the sector.
        """
        return self.core_model.calc_dim_su2_sector(total_s)
