import edu1sces.core
from .types import Site, Bond
from .utils import (
    build_site_index,
    convert_onsite_potential_to_array,
    convert_intersite_interactions_to_indexed_dict,
)


class KondoLattice2ChModel:
    def __init__(
        self,
        spins: dict[Site, float],
        hopping_0: dict[Bond, float] | None = None,
        hopping_1: dict[Bond, float] | None = None,
        u_0: dict[Site, float] | None = None,
        u_1: dict[Site, float] | None = None,
        mu_0: dict[Site, float] | None = None,
        mu_1: dict[Site, float] | None = None,
        hz_c_0: dict[Site, float] | None = None,
        hz_c_1: dict[Site, float] | None = None,
        density_density_0: dict[Bond, float] | None = None,
        density_density_1: dict[Bond, float] | None = None,
        density_density_01: dict[Bond, float] | None = None,
        hz_f: dict[Site, float] | None = None,
        d: dict[Site, float] | None = None,
        ff_exchange_xy: dict[Bond, float] | None = None,
        ff_exchange_z: dict[Bond, float] | None = None,
        kondo_xy_0: dict[Site, float] | None = None,
        kondo_z_0: dict[Site, float] | None = None,
        kondo_xy_1: dict[Site, float] | None = None,
        kondo_z_1: dict[Site, float] | None = None,
    ) -> None:
        if hopping_0 is None:
            hopping_0 = {}
        if hopping_1 is None:
            hopping_1 = {}
        if u_0 is None:
            u_0 = {}
        if u_1 is None:
            u_1 = {}
        if mu_0 is None:
            mu_0 = {}
        if mu_1 is None:
            mu_1 = {}
        if hz_c_0 is None:
            hz_c_0 = {}
        if hz_c_1 is None:
            hz_c_1 = {}
        if density_density_0 is None:
            density_density_0 = {}
        if density_density_1 is None:
            density_density_1 = {}
        if density_density_01 is None:
            density_density_01 = {}
        if hz_f is None:
            hz_f = {}
        if d is None:
            d = {}
        if ff_exchange_xy is None:
            ff_exchange_xy = {}
        if ff_exchange_z is None:
            ff_exchange_z = {}
        if kondo_xy_0 is None:
            kondo_xy_0 = {}
        if kondo_z_0 is None:
            kondo_z_0 = {}
        if kondo_xy_1 is None:
            kondo_xy_1 = {}
        if kondo_z_1 is None:
            kondo_z_1 = {}

        self.site_to_integer = build_site_index(list(spins.keys()))
        self.num_sites = len(self.site_to_integer)
        if self.num_sites == 0:
            raise ValueError("spins must be non-empty")

        self.spin_list = convert_onsite_potential_to_array(self.site_to_integer, spins)
        self.u_list_0 = convert_onsite_potential_to_array(self.site_to_integer, u_0)
        self.u_list_1 = convert_onsite_potential_to_array(self.site_to_integer, u_1)
        self.mu_list_0 = convert_onsite_potential_to_array(self.site_to_integer, mu_0)
        self.mu_list_1 = convert_onsite_potential_to_array(self.site_to_integer, mu_1)
        self.hz_c_list_0 = convert_onsite_potential_to_array(
            self.site_to_integer, hz_c_0
        )
        self.hz_c_list_1 = convert_onsite_potential_to_array(
            self.site_to_integer, hz_c_1
        )
        self.hz_f_list = convert_onsite_potential_to_array(self.site_to_integer, hz_f)
        self.d_list = convert_onsite_potential_to_array(self.site_to_integer, d)
        self.kondo_xy_list_0 = convert_onsite_potential_to_array(
            self.site_to_integer, kondo_xy_0
        )
        self.kondo_z_list_0 = convert_onsite_potential_to_array(
            self.site_to_integer, kondo_z_0
        )
        self.kondo_xy_list_1 = convert_onsite_potential_to_array(
            self.site_to_integer, kondo_xy_1
        )
        self.kondo_z_list_1 = convert_onsite_potential_to_array(
            self.site_to_integer, kondo_z_1
        )

        self.hopping_0 = convert_intersite_interactions_to_indexed_dict(
            self.site_to_integer, hopping_0, directed=True
        )
        self.hopping_1 = convert_intersite_interactions_to_indexed_dict(
            self.site_to_integer, hopping_1, directed=True
        )

        self.density_density_0 = convert_intersite_interactions_to_indexed_dict(
            self.site_to_integer, density_density_0, directed=False
        )
        self.density_density_1 = convert_intersite_interactions_to_indexed_dict(
            self.site_to_integer, density_density_1, directed=False
        )
        self.density_density_01 = convert_intersite_interactions_to_indexed_dict(
            self.site_to_integer, density_density_01, directed=False
        )

        self.ff_exchange_xy = convert_intersite_interactions_to_indexed_dict(
            self.site_to_integer, ff_exchange_xy, directed=False
        )
        self.ff_exchange_z = convert_intersite_interactions_to_indexed_dict(
            self.site_to_integer, ff_exchange_z, directed=False
        )

        self.core_model = edu1sces.core.KondoLattice2ChModel(
            self.spin_list,
            self.hopping_0,
            self.hopping_1,
            self.u_list_0,
            self.u_list_1,
            self.mu_list_0,
            self.mu_list_1,
            self.hz_c_list_0,
            self.hz_c_list_1,
            self.density_density_0,
            self.density_density_1,
            self.density_density_01,
            self.hz_f_list,
            self.d_list,
            self.ff_exchange_xy,
            self.ff_exchange_z,
            self.kondo_xy_list_0,
            self.kondo_z_list_0,
            self.kondo_xy_list_1,
            self.kondo_z_list_1,
        )
