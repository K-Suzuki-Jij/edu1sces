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
            total_sz: Total Sz value of the sector (integer or half-integer).

        Returns:
            Dimension of the sector.
        """
        return self.core_model.calc_dim_u1_sector(total_sz)

    def make_local_op_sz(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local z-component spin operator for a given site.

        For a spin-S site, returns a (2S+1) x (2S+1) diagonal matrix with
        eigenvalues S, S-1, ..., -S+1, -S (from top to bottom).

        .. math::

            S_z |S, m\rangle = m |S, m\rangle

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local Sz operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_sz(site_index)

    def make_local_op_sp(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin raising operator for a given site.

        For a spin-S site, returns a (2S+1) x (2S+1) matrix.

        .. math::

            S^+ |S, m\rangle = \sqrt{S(S+1) - m(m+1)} |S, m+1\rangle

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local S+ operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_sp(site_index)

    def make_local_op_sm(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin lowering operator for a given site.

        For a spin-S site, returns a (2S+1) x (2S+1) matrix.

        .. math::

            S^- = (S^+)^\dagger

            S^- |S, m\rangle = \sqrt{S(S+1) - m(m-1)} |S, m-1\rangle

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local S- operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_sm(site_index)

    def make_local_op_sx(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local x-component spin operator for a given site.

        .. math::

            S_x = \frac{1}{2}(S^+ + S^-)

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local Sx operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_sx(site_index)

    def make_local_op_isy(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create i times the local y-component spin operator for a given site.

        Returns i*Sy instead of Sy to avoid complex numbers in the matrix.

        .. math::

            i S_y = \frac{1}{2}(S^+ - S^-)

        Note: To get the expectation value of Sy, divide by i (or multiply by -i).

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local i*Sy operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_isy(site_index)

    def make_local_hamiltonian(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local (on-site) Hamiltonian for a given site.

        The local Hamiltonian includes the Zeeman field and single-ion
        anisotropy terms:

        .. math::

            H_i = h^z_i S^z_i + D_i (S^z_i)^2

        where:
        - h^z_i: Zeeman field along z-axis
        - D_i: Single-ion anisotropy

        Args:
            site: The site for which to construct the local Hamiltonian.

        Returns:
            Local Hamiltonian as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_hamiltonian(site_index)
