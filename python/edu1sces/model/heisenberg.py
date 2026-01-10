import edu1sces.core
from .utils import (
    build_site_index,
    convert_onsite_potential_to_array,
    convert_intersite_interactions_to_indexed_dict,
)
from .types import Site, Bond


class HeisenbergModel:
    def __init__(
        self,
        spins: dict[Site, float],
        exchange_xy: dict[Bond, float] | None = None,
        exchange_z: dict[Bond, float] | None = None,
        hz: dict[Site, float] | None = None,
        d: dict[Site, float] | None = None,
    ) -> None:
        if exchange_xy is None:
            exchange_xy = {}
        if exchange_z is None:
            exchange_z = {}
        if hz is None:
            hz = {}
        if d is None:
            d = {}

        self.site_to_integer = build_site_index(spins.keys())
        self.num_sites = len(self.site_to_integer)
        if self.num_sites == 0:
            raise ValueError("site_list must be non-empty")

        self.spin_list = convert_onsite_potential_to_array(self.site_to_integer, spins)
        self.hz_list = convert_onsite_potential_to_array(self.site_to_integer, hz)
        self.d_list = convert_onsite_potential_to_array(self.site_to_integer, d)

        self.exchange_xy = convert_intersite_interactions_to_indexed_dict(
            self.site_to_integer, exchange_xy, directed=False
        )
        self.exchange_z = convert_intersite_interactions_to_indexed_dict(
            self.site_to_integer, exchange_z, directed=False
        )

        self.core_model = edu1sces.core.HeisenbergModel(
            self.spin_list,
            self.hz_list,
            self.d_list,
            self.exchange_xy,
            self.exchange_z,
        )

    def calc_dim_u1_sector(self, total_sz: float) -> int:
        """Calculate the dimension of the U(1) sector with the given total Sz.

        Args:
            total_sz (float): Total Sz value of the sector.

        Returns:
            int: Dimension of the sector.
        """
        return self.core_model.calc_dim_u1_sector(total_sz)
