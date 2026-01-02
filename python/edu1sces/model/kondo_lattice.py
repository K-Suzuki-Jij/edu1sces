import edu1sces.core
from .types import Site, Bond
from .utils import (
    build_site_index,
    convert_onsite_potential_to_array,
    convert_intersite_interactions_to_indexed_dict,
)


class KondoLatticeModel:
    def __init__(
        self,
        spins: dict[Site, float],
        hopping: dict[Bond, float] | None = None,
        u: dict[Site, float] | None = None,
        mu: dict[Site, float] | None = None,
        hz_c: dict[Site, float] | None = None,
        hz_f: dict[Site, float] | None = None,
        d: dict[Site, float] | None = None,
        density_density: dict[Bond, float] | None = None,
        kondo_xy: dict[Site, float] | None = None,
        kondo_z: dict[Site, float] | None = None,
        ff_exchange_xy: dict[Bond, float] | None = None,
        ff_exchange_z: dict[Bond, float] | None = None,
    ) -> None:
        if hopping is None:
            hopping = {}
        if u is None:
            u = {}
        if mu is None:
            mu = {}
        if hz_c is None:
            hz_c = {}
        if hz_f is None:
            hz_f = {}
        if d is None:
            d = {}
        if density_density is None:
            density_density = {}
        if kondo_xy is None:
            kondo_xy = {}
        if kondo_z is None:
            kondo_z = {}
        if ff_exchange_xy is None:
            ff_exchange_xy = {}
        if ff_exchange_z is None:
            ff_exchange_z = {}

        self.site_to_integer = build_site_index(list(spins.keys()))
        self.num_sites = len(self.site_to_integer)
        if self.num_sites == 0:
            raise ValueError("spins must be non-empty")

        self.spin_list = convert_onsite_potential_to_array(self.site_to_integer, spins)
        self.u_list = convert_onsite_potential_to_array(self.site_to_integer, u)
        self.mu_list = convert_onsite_potential_to_array(self.site_to_integer, mu)
        self.hz_c_list = convert_onsite_potential_to_array(self.site_to_integer, hz_c)
        self.hz_f_list = convert_onsite_potential_to_array(self.site_to_integer, hz_f)
        self.d_list = convert_onsite_potential_to_array(self.site_to_integer, d)
        self.kondo_xy_list = convert_onsite_potential_to_array(
            self.site_to_integer, kondo_xy
        )
        self.kondo_z_list = convert_onsite_potential_to_array(
            self.site_to_integer, kondo_z
        )

        self.hopping = convert_intersite_interactions_to_indexed_dict(
            self.site_to_integer,
            hopping,
            directed=True,
        )
        self.density_density = convert_intersite_interactions_to_indexed_dict(
            self.site_to_integer,
            density_density,
            directed=False,
        )
        self.ff_exchange_xy = convert_intersite_interactions_to_indexed_dict(
            self.site_to_integer,
            ff_exchange_xy,
            directed=False,
        )
        self.ff_exchange_z = convert_intersite_interactions_to_indexed_dict(
            self.site_to_integer,
            ff_exchange_z,
            directed=False,
        )

        self.core_model = edu1sces.core.KondoLatticeModel(
            self.spin_list,
            self.hopping,
            self.u_list,
            self.mu_list,
            self.hz_c_list,
            self.hz_f_list,
            self.d_list,
            self.density_density,
            self.kondo_xy_list,
            self.kondo_z_list,
            self.ff_exchange_xy,
            self.ff_exchange_z,
        )
