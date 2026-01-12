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

    def calc_dim_u1_sector(self, num_electrons: int, total_sz: float) -> int:
        """Calculate the dimension of the U(1) sector with the given quantum numbers.

        The Hubbard model conserves both the total particle number N and
        the total z-component of spin Sz. This method computes the dimension
        of the Hilbert space sector specified by these quantum numbers.

        Args:
            num_electrons: Total number of electrons N.
            total_sz: Total z-component of spin Sz (integer or half-integer).

        Returns:
            Dimension of the (N, Sz) sector.
        """
        return self.core_model.calc_dim_u1_sector(num_electrons, total_sz)

    def make_local_op_c_up(self) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin-up electron annihilation operator.

        Returns the 4x4 matrix representation of c_up in the local Fock space
        with basis ordering: |vac>, |up>, |down>, |updown>.

        .. math::

            c_\uparrow = \begin{pmatrix}
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 \\
                0 & 0 & 0 & 0
            \end{pmatrix}

        Note: The sign on the |updown> -> |down> transition accounts for the
        fermion anticommutation relation (down electron comes first).

        Returns:
            Local spin-up annihilation operator as a CSR matrix.
        """
        return self.core_model.make_local_op_c_up()

    def make_local_op_c_down(self) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin-down electron annihilation operator.

        Returns the 4x4 matrix representation of c_down in the local Fock space
        with basis ordering: |vac>, |up>, |down>, |updown>.

        .. math::

            c_\downarrow = \begin{pmatrix}
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & -1 \\
                0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0
            \end{pmatrix}

        Note: The sign on the |updown> -> |up> transition accounts for the
        fermion anticommutation relation (need to move past the up electron).

        Returns:
            Local spin-down annihilation operator as a CSR matrix.
        """
        return self.core_model.make_local_op_c_down()

    def make_local_op_c_up_dag(self) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin-up electron creation operator.

        Returns the Hermitian conjugate of c_up:

        .. math::

            c_\uparrow^\dagger = (c_\uparrow)^T

        Returns:
            Local spin-up creation operator as a CSR matrix.
        """
        return self.core_model.make_local_op_c_up_dag()

    def make_local_op_c_down_dag(self) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin-down electron creation operator.

        Returns the Hermitian conjugate of c_down:

        .. math::

            c_\downarrow^\dagger = (c_\downarrow)^T

        Returns:
            Local spin-down creation operator as a CSR matrix.
        """
        return self.core_model.make_local_op_c_down_dag()

    def make_local_op_n_up(self) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin-up number operator.

        .. math::

            n_\uparrow = c_\uparrow^\dagger c_\uparrow

        Diagonal matrix with eigenvalues 0 or 1 indicating the presence
        of a spin-up electron.

        Returns:
            Local spin-up number operator as a CSR matrix.
        """
        return self.core_model.make_local_op_n_up()

    def make_local_op_n_down(self) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin-down number operator.

        .. math::

            n_\downarrow = c_\downarrow^\dagger c_\downarrow

        Diagonal matrix with eigenvalues 0 or 1 indicating the presence
        of a spin-down electron.

        Returns:
            Local spin-down number operator as a CSR matrix.
        """
        return self.core_model.make_local_op_n_down()

    def make_local_op_n(self) -> edu1sces.core.CsrMatrix:
        r"""Create the local total number operator.

        .. math::

            n = n_\uparrow + n_\downarrow

        Diagonal matrix with eigenvalues 0, 1, or 2 indicating the total
        number of electrons at the site.

        Returns:
            Local total number operator as a CSR matrix.
        """
        return self.core_model.make_local_op_n()

    def make_local_op_sz(self) -> edu1sces.core.CsrMatrix:
        r"""Create the local z-component spin operator.

        .. math::

            S_z = \frac{1}{2}(n_\uparrow - n_\downarrow)

        Returns:
            Local Sz operator as a CSR matrix.
        """
        return self.core_model.make_local_op_sz()

    def make_local_op_sp(self) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin raising operator.

        .. math::

            S^+ = c_\uparrow^\dagger c_\downarrow

        Raises the spin by flipping a down electron to up.

        Returns:
            Local S+ operator as a CSR matrix.
        """
        return self.core_model.make_local_op_sp()

    def make_local_op_sm(self) -> edu1sces.core.CsrMatrix:
        r"""Create the local spin lowering operator.

        .. math::

            S^- = c_\downarrow^\dagger c_\uparrow = (S^+)^\dagger

        Lowers the spin by flipping an up electron to down.

        Returns:
            Local S- operator as a CSR matrix.
        """
        return self.core_model.make_local_op_sm()

    def make_local_hamiltonian(self, site: Site) -> edu1sces.core.CsrMatrix:
        r"""Create the local (on-site) Hamiltonian for a given site.

        The local Hamiltonian includes the on-site Coulomb repulsion,
        chemical potential, and Zeeman field terms:

        .. math::

            H_i = U_i \, n_{i\uparrow} n_{i\downarrow}
                  - \mu_i \, n_i
                  + h^z_i \, S^z_i

        where:
        - U_i: On-site Coulomb repulsion
        - mu_i: Chemical potential
        - h^z_i: Zeeman field along z-axis

        Args:
            site: The site for which to construct the local Hamiltonian.

        Returns:
            Local Hamiltonian as a CSR matrix.
        """
        site_index = self.site_to_integer[site]
        return self.core_model.make_local_hamiltonian(site_index)
