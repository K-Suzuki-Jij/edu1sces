import pytest
from edu1sces.model import KondoLatticeModel


def test_minimal_spins_only() -> None:
    km = KondoLatticeModel(
        spins={0: 0.5, 1: 1.0},
    )
    assert km.num_sites == 2
    assert km.spin_list == [0.5, 1.0]
    assert km.u_list == [0.0, 0.0]
    assert km.mu_list == [0.0, 0.0]
    assert km.hz_c_list == [0.0, 0.0]
    assert km.hz_f_list == [0.0, 0.0]
    assert km.d_list == [0.0, 0.0]
    assert km.kondo_xy_list == [0.0, 0.0]
    assert km.kondo_z_list == [0.0, 0.0]
    assert km.hopping == {}
    assert km.density_density == {}
    assert km.ff_exchange_xy == {}
    assert km.ff_exchange_z == {}

    assert km.core_model.num_sites == km.num_sites
    assert km.core_model.two_s_list == [1, 2]
    assert km.core_model.hopping == km.hopping
    assert km.core_model.u_list == km.u_list
    assert km.core_model.mu_list == km.mu_list
    assert km.core_model.hz_c_list == km.hz_c_list
    assert km.core_model.hz_f_list == km.hz_f_list
    assert km.core_model.d_list == km.d_list
    assert km.core_model.density_density == km.density_density
    assert km.core_model.kondo_xy_list == km.kondo_xy_list
    assert km.core_model.kondo_z_list == km.kondo_z_list
    assert km.core_model.ff_exchange_xy == km.ff_exchange_xy
    assert km.core_model.ff_exchange_z == km.ff_exchange_z


def test_site_union_from_all_inputs() -> None:
    n = 4
    spins = {i: 0.5 for i in range(n)}

    km = KondoLatticeModel(
        spins=spins,
        hopping={(i, i + 1): 1.0 for i in range(n - 1)},
        u={i: 2.0 for i in range(n)},
        mu={i: -0.1 * i for i in range(n)},
        hz_c={0: 0.25},
        hz_f={1: 0.5},
        d={2: 1.0},
        density_density={(0, 1): 1.25, (1, 0): 2.5},
        kondo_xy={3: 2.0},
        kondo_z={0: 3.0},
        ff_exchange_xy={(0, 1): 1.25, (1, 0): 2.5},
        ff_exchange_z={(0, 1): 1.25, (1, 0): 2.5},
    )

    assert km.num_sites == n
    assert km.spin_list == [0.5] * n
    assert km.u_list == [2.0] * n
    assert km.mu_list == pytest.approx([0.0, -0.1, -0.2, -0.3])
    assert km.hz_c_list == [0.25, 0.0, 0.0, 0.0]
    assert km.hz_f_list == [0.0, 0.5, 0.0, 0.0]
    assert km.d_list == [0.0, 0.0, 1.0, 0.0]
    assert km.kondo_xy_list == [0.0, 0.0, 0.0, 2.0]
    assert km.kondo_z_list == [3.0, 0.0, 0.0, 0.0]

    assert km.hopping == {(i, i + 1): 1.0 for i in range(n - 1)}
    assert km.density_density == {(0, 1): 3.75}
    assert km.ff_exchange_xy == {(0, 1): 3.75}
    assert km.ff_exchange_z == {(0, 1): 3.75}

    assert km.core_model.num_sites == km.num_sites
    assert km.core_model.two_s_list == [1] * n
    assert km.core_model.hopping == km.hopping
    assert km.core_model.u_list == km.u_list
    assert km.core_model.mu_list == km.mu_list
    assert km.core_model.hz_c_list == km.hz_c_list
    assert km.core_model.hz_f_list == km.hz_f_list
    assert km.core_model.d_list == km.d_list
    assert km.core_model.density_density == km.density_density
    assert km.core_model.kondo_xy_list == km.kondo_xy_list
    assert km.core_model.kondo_z_list == km.kondo_z_list
    assert km.core_model.ff_exchange_xy == km.ff_exchange_xy
    assert km.core_model.ff_exchange_z == km.ff_exchange_z


def test_directed_hopping_preserves_both_directions() -> None:
    km = KondoLatticeModel(
        spins={0: 0.5, 1: 0.5},
        hopping={(0, 1): 1.0, (1, 0): 2.0},
    )
    assert km.num_sites == 2
    assert km.spin_list == [0.5, 0.5]
    assert km.u_list == [0.0, 0.0]
    assert km.mu_list == [0.0, 0.0]
    assert km.hz_c_list == [0.0, 0.0]
    assert km.hz_f_list == [0.0, 0.0]
    assert km.d_list == [0.0, 0.0]
    assert km.kondo_xy_list == [0.0, 0.0]
    assert km.kondo_z_list == [0.0, 0.0]
    assert km.hopping == {(0, 1): 1.0, (1, 0): 2.0}
    assert km.density_density == {}
    assert km.ff_exchange_xy == {}
    assert km.ff_exchange_z == {}

    assert km.core_model.num_sites == km.num_sites
    assert km.core_model.two_s_list == [1, 1]
    assert km.core_model.hopping == km.hopping
    assert km.core_model.u_list == km.u_list
    assert km.core_model.mu_list == km.mu_list
    assert km.core_model.hz_c_list == km.hz_c_list
    assert km.core_model.hz_f_list == km.hz_f_list
    assert km.core_model.d_list == km.d_list
    assert km.core_model.density_density == km.density_density
    assert km.core_model.kondo_xy_list == km.kondo_xy_list
    assert km.core_model.kondo_z_list == km.kondo_z_list
    assert km.core_model.ff_exchange_xy == km.ff_exchange_xy
    assert km.core_model.ff_exchange_z == km.ff_exchange_z


def test_undirected_density_density_merges_and_sums() -> None:
    km = KondoLatticeModel(
        spins={0: 0.5, 1: 0.5},
        density_density={(0, 1): 1.25, (1, 0): 2.5},
    )
    assert km.num_sites == 2
    assert km.spin_list == [0.5, 0.5]
    assert km.u_list == [0.0, 0.0]
    assert km.mu_list == [0.0, 0.0]
    assert km.hz_c_list == [0.0, 0.0]
    assert km.hz_f_list == [0.0, 0.0]
    assert km.d_list == [0.0, 0.0]
    assert km.kondo_xy_list == [0.0, 0.0]
    assert km.kondo_z_list == [0.0, 0.0]
    assert km.hopping == {}
    assert km.density_density == {(0, 1): 3.75}
    assert km.ff_exchange_xy == {}
    assert km.ff_exchange_z == {}

    assert km.core_model.num_sites == km.num_sites
    assert km.core_model.two_s_list == [1, 1]
    assert km.core_model.hopping == km.hopping
    assert km.core_model.u_list == km.u_list
    assert km.core_model.mu_list == km.mu_list
    assert km.core_model.hz_c_list == km.hz_c_list
    assert km.core_model.hz_f_list == km.hz_f_list
    assert km.core_model.d_list == km.d_list
    assert km.core_model.density_density == km.density_density
    assert km.core_model.kondo_xy_list == km.kondo_xy_list
    assert km.core_model.kondo_z_list == km.kondo_z_list
    assert km.core_model.ff_exchange_xy == km.ff_exchange_xy
    assert km.core_model.ff_exchange_z == km.ff_exchange_z


def test_undirected_ff_exchange_xy_merges_and_sums() -> None:
    km = KondoLatticeModel(
        spins={0: 0.5, 1: 0.5},
        ff_exchange_xy={(0, 1): 1.25, (1, 0): 2.5},
    )
    assert km.num_sites == 2
    assert km.spin_list == [0.5, 0.5]
    assert km.u_list == [0.0, 0.0]
    assert km.mu_list == [0.0, 0.0]
    assert km.hz_c_list == [0.0, 0.0]
    assert km.hz_f_list == [0.0, 0.0]
    assert km.d_list == [0.0, 0.0]
    assert km.kondo_xy_list == [0.0, 0.0]
    assert km.kondo_z_list == [0.0, 0.0]
    assert km.hopping == {}
    assert km.density_density == {}
    assert km.ff_exchange_xy == {(0, 1): 3.75}
    assert km.ff_exchange_z == {}

    assert km.core_model.num_sites == km.num_sites
    assert km.core_model.two_s_list == [1, 1]
    assert km.core_model.hopping == km.hopping
    assert km.core_model.u_list == km.u_list
    assert km.core_model.mu_list == km.mu_list
    assert km.core_model.hz_c_list == km.hz_c_list
    assert km.core_model.hz_f_list == km.hz_f_list
    assert km.core_model.d_list == km.d_list
    assert km.core_model.density_density == km.density_density
    assert km.core_model.kondo_xy_list == km.kondo_xy_list
    assert km.core_model.kondo_z_list == km.kondo_z_list
    assert km.core_model.ff_exchange_xy == km.ff_exchange_xy
    assert km.core_model.ff_exchange_z == km.ff_exchange_z


def test_undirected_ff_exchange_z_merges_and_sums() -> None:
    km = KondoLatticeModel(
        spins={0: 0.5, 1: 0.5},
        ff_exchange_z={(0, 1): 1.25, (1, 0): 2.5},
    )
    assert km.num_sites == 2
    assert km.spin_list == [0.5, 0.5]
    assert km.u_list == [0.0, 0.0]
    assert km.mu_list == [0.0, 0.0]
    assert km.hz_c_list == [0.0, 0.0]
    assert km.hz_f_list == [0.0, 0.0]
    assert km.d_list == [0.0, 0.0]
    assert km.kondo_xy_list == [0.0, 0.0]
    assert km.kondo_z_list == [0.0, 0.0]
    assert km.hopping == {}
    assert km.density_density == {}
    assert km.ff_exchange_xy == {}
    assert km.ff_exchange_z == {(0, 1): 3.75}

    assert km.core_model.num_sites == km.num_sites
    assert km.core_model.two_s_list == [1, 1]
    assert km.core_model.hopping == km.hopping
    assert km.core_model.u_list == km.u_list
    assert km.core_model.mu_list == km.mu_list
    assert km.core_model.hz_c_list == km.hz_c_list
    assert km.core_model.hz_f_list == km.hz_f_list
    assert km.core_model.d_list == km.d_list
    assert km.core_model.density_density == km.density_density
    assert km.core_model.kondo_xy_list == km.kondo_xy_list
    assert km.core_model.kondo_z_list == km.kondo_z_list
    assert km.core_model.ff_exchange_xy == km.ff_exchange_xy
    assert km.core_model.ff_exchange_z == km.ff_exchange_z


def test_missing_spin_for_bond_site_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        KondoLatticeModel(
            spins={0: 0.5},
            hopping={(0, 1): 1.0},
        )


def test_empty_spins_raises() -> None:
    with pytest.raises(ValueError):
        KondoLatticeModel(
            spins={},
        )
