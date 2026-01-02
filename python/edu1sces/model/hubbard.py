import edu1sces.core
from .types import Site, Bond
from .utils import (
    build_site_index,
    convert_onsite_potential_to_array,
    convert_intersite_interactions_to_indexed_dict,
    collect_sites_from_bonds,
)


class HubbardModel:
    def __init__(
        self,
        hopping: dict[Bond, float] | None = None,
        u: dict[Site, float] | None = None,
        mu: dict[Site, float] | None = None,
        hz: dict[Site, float] | None = None,
        density_density: dict[Bond, float] | None = None,
        exchange_xy: dict[Bond, float] | None = None,
        exchange_z: dict[Bond, float] | None = None,
    ) -> None:
        if hopping is None:
            hopping = {}
        if u is None:
            u = {}
        if mu is None:
            mu = {}
        if hz is None:
            hz = {}
        if density_density is None:
            density_density = {}
        if exchange_xy is None:
            exchange_xy = {}
        if exchange_z is None:
            exchange_z = {}

        site_set: set[Site] = set()
        site_set.update(u.keys())
        site_set.update(mu.keys())
        site_set.update(hz.keys())
        site_set.update(collect_sites_from_bonds(hopping))
        site_set.update(collect_sites_from_bonds(density_density))
        site_set.update(collect_sites_from_bonds(exchange_xy))
        site_set.update(collect_sites_from_bonds(exchange_z))

        if not site_set:
            raise ValueError("at least one site must be specified")

        self.site_to_integer = build_site_index(list(site_set))
        self.num_sites = len(self.site_to_integer)

        self.u_list = convert_onsite_potential_to_array(self.site_to_integer, u)
        self.mu_list = convert_onsite_potential_to_array(self.site_to_integer, mu)
        self.hz_list = convert_onsite_potential_to_array(self.site_to_integer, hz)

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
        self.exchange_xy = convert_intersite_interactions_to_indexed_dict(
            self.site_to_integer,
            exchange_xy,
            directed=False,
        )
        self.exchange_z = convert_intersite_interactions_to_indexed_dict(
            self.site_to_integer,
            exchange_z,
            directed=False,
        )

        self.core_model = edu1sces.core.HubbardModel(
            self.hopping,
            self.u_list,
            self.mu_list,
            self.hz_list,
            self.density_density,
            self.exchange_xy,
            self.exchange_z,
        )
