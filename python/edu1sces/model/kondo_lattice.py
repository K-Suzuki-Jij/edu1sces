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

    def calc_dim_u1_sector(self, num_electrons: int, total_sz: float) -> int:
        """Calculate the dimension of the U(1) sector with the given quantum numbers.

        The Kondo lattice model conserves both the total particle number N and
        the total z-component of spin Sz (sum of conduction electron and localized
        spin contributions). This method computes the dimension of the Hilbert
        space sector specified by these quantum numbers.

        Args:
            num_electrons: Total number of conduction electrons N.
            total_sz: Total z-component of spin Sz (integer or half-integer).

        Returns:
            Dimension of the (N, Sz) sector.
        """
        return self.core_model.calc_dim_u1_sector(num_electrons, total_sz)

    def make_local_op_c_up(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin-up electron annihilation operator for a given site.

        Returns the matrix representation of c_up in the local Fock space.
        The local basis is ordered as:

        .. math::

            |S\rangle|vac\rangle, |S\rangle|\uparrow\rangle,
            |S\rangle|\downarrow\rangle, |S\rangle|\uparrow\downarrow\rangle,
            |S-1\rangle|vac\rangle, \ldots

        where the first ket is the localized spin state and the second is
        the conduction electron state.

        The operator acts as:

        .. math::

            c_\uparrow |m\rangle|\uparrow\rangle = |m\rangle|vac\rangle

            c_\uparrow |m\rangle|\uparrow\downarrow\rangle = |m\rangle|\downarrow\rangle

        Note: The sign on the |updown> -> |down> transition is +1 because
        up electron comes first in our ordering.

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local spin-up annihilation operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_c_up(site_index)

    def make_local_op_c_down(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin-down electron annihilation operator for a given site.

        Returns the matrix representation of c_down in the local Fock space.
        The local basis is ordered as:

        .. math::

            |S\rangle|vac\rangle, |S\rangle|\uparrow\rangle,
            |S\rangle|\downarrow\rangle, |S\rangle|\uparrow\downarrow\rangle,
            |S-1\rangle|vac\rangle, \ldots

        where the first ket is the localized spin state and the second is
        the conduction electron state.

        The operator acts as:

        .. math::

            c_\downarrow |m\rangle|\downarrow\rangle = |m\rangle|vac\rangle

            c_\downarrow |m\rangle|\uparrow\downarrow\rangle = -|m\rangle|\uparrow\rangle

        Note: The sign on the |updown> -> |up> transition is -1 because
        down electron must pass the up electron.

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local spin-down annihilation operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_c_down(site_index)

    def make_local_op_c_up_dag(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin-up electron creation operator for a given site.

        Returns the Hermitian conjugate of c_up:

        .. math::

            c_\uparrow^\dagger = (c_\uparrow)^T

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local spin-up creation operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_c_up_dag(site_index)

    def make_local_op_c_down_dag(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin-down electron creation operator for a given site.

        Returns the Hermitian conjugate of c_down:

        .. math::

            c_\downarrow^\dagger = (c_\downarrow)^T

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local spin-down creation operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_c_down_dag(site_index)

    def make_local_op_n_up(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin-up number operator for a given site.

        .. math::

            n_\uparrow = c_\uparrow^\dagger c_\uparrow

        Diagonal matrix with eigenvalues 0 or 1 indicating the presence
        of a spin-up electron.

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local spin-up number operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_n_up(site_index)

    def make_local_op_n_down(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin-down number operator for a given site.

        .. math::

            n_\downarrow = c_\downarrow^\dagger c_\downarrow

        Diagonal matrix with eigenvalues 0 or 1 indicating the presence
        of a spin-down electron.

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local spin-down number operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_n_down(site_index)

    def make_local_op_n(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local total conduction electron number operator.

        .. math::

            n = n_\uparrow + n_\downarrow

        Diagonal matrix with eigenvalues 0, 1, or 2 indicating the total
        number of conduction electrons at the site.

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local total number operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_n(site_index)

    def make_local_op_c_sz(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local z-component spin operator for conduction electrons.

        .. math::

            s_z = \frac{1}{2}(n_\uparrow - n_\downarrow)

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local conduction electron Sz operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_c_sz(site_index)

    def make_local_op_c_sp(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin raising operator for conduction electrons.

        .. math::

            s^+ = c_\uparrow^\dagger c_\downarrow

        Raises the spin by flipping a down electron to up.

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local conduction electron S+ operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_c_sp(site_index)

    def make_local_op_c_sm(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin lowering operator for conduction electrons.

        .. math::

            s^- = c_\downarrow^\dagger c_\uparrow = (s^+)^\dagger

        Lowers the spin by flipping an up electron to down.

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local conduction electron S- operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_c_sm(site_index)

    def make_local_op_l_sz(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local z-component spin operator for localized spins.

        For a spin-S site, this is a diagonal operator with eigenvalues
        S, S-1, ..., -S+1, -S (repeated for each electron state).

        .. math::

            S_z |m\rangle|e\rangle = m |m\rangle|e\rangle

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local localized spin Sz operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_l_sz(site_index)

    def make_local_op_l_sp(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin raising operator for localized spins.

        .. math::

            S^+ |m\rangle|e\rangle = \sqrt{S(S+1) - m(m+1)} |m+1\rangle|e\rangle

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local localized spin S+ operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_l_sp(site_index)

    def make_local_op_l_sm(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin lowering operator for localized spins.

        .. math::

            S^- = (S^+)^\dagger

            S^- |m\rangle|e\rangle = \sqrt{S(S+1) - m(m-1)} |m-1\rangle|e\rangle

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local localized spin S- operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_l_sm(site_index)

    def make_local_op_l_sx(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local x-component spin operator for localized spins.

        .. math::

            S_x = \frac{1}{2}(S^+ + S^-)

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local localized spin Sx operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_l_sx(site_index)

    def make_local_op_l_isy(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create i times the local y-component spin operator for localized spins.

        Returns i*Sy instead of Sy to avoid complex numbers in the matrix.

        .. math::

            i S_y = \frac{1}{2}(S^+ - S^-)

        Note: To get the expectation value of Sy, divide by i (or multiply by -i).

        Args:
            site: The site for which to construct the operator.

        Returns:
            Local localized spin i*Sy operator as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_op_l_isy(site_index)
