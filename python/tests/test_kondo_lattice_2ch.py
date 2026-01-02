import pytest
from edu1sces.model import KondoLattice2ChModel


def test_minimal_spins_only() -> None:
    km = KondoLattice2ChModel(
        spins={0: 0.5, 1: 1.0},
    )
    assert km.num_sites == 2
    assert km.spin_list == [0.5, 1.0]
    assert km.u_list_0 == [0.0, 0.0]
    assert km.u_list_1 == [0.0, 0.0]
    assert km.mu_list_0 == [0.0, 0.0]
    assert km.mu_list_1 == [0.0, 0.0]
    assert km.hz_c_list_0 == [0.0, 0.0]
    assert km.hz_c_list_1 == [0.0, 0.0]
    assert km.hz_f_list == [0.0, 0.0]
    assert km.d_list == [0.0, 0.0]
    assert km.kondo_xy_list_0 == [0.0, 0.0]
    assert km.kondo_z_list_0 == [0.0, 0.0]
    assert km.kondo_xy_list_1 == [0.0, 0.0]
    assert km.kondo_z_list_1 == [0.0, 0.0]

    assert km.hopping_0 == {}
    assert km.hopping_1 == {}
    assert km.density_density_0 == {}
    assert km.density_density_1 == {}
    assert km.density_density_01 == {}
    assert km.ff_exchange_xy == {}
    assert km.ff_exchange_z == {}

    assert km.core_model.num_sites == km.num_sites
    assert km.core_model.two_s_list == [1, 2]
    assert km.core_model.hopping_0 == km.hopping_0
    assert km.core_model.hopping_1 == km.hopping_1
    assert km.core_model.u_list_0 == km.u_list_0
    assert km.core_model.u_list_1 == km.u_list_1
    assert km.core_model.mu_list_0 == km.mu_list_0
    assert km.core_model.mu_list_1 == km.mu_list_1
    assert km.core_model.hz_c_list_0 == km.hz_c_list_0
    assert km.core_model.hz_c_list_1 == km.hz_c_list_1
    assert km.core_model.density_density_0 == km.density_density_0
    assert km.core_model.density_density_1 == km.density_density_1
    assert km.core_model.density_density_01 == km.density_density_01
    assert km.core_model.hz_f_list == km.hz_f_list
    assert km.core_model.d_list == km.d_list
    assert km.core_model.ff_exchange_xy == km.ff_exchange_xy
    assert km.core_model.ff_exchange_z == km.ff_exchange_z
    assert km.core_model.kondo_xy_list_0 == km.kondo_xy_list_0
    assert km.core_model.kondo_z_list_0 == km.kondo_z_list_0
    assert km.core_model.kondo_xy_list_1 == km.kondo_xy_list_1
    assert km.core_model.kondo_z_list_1 == km.kondo_z_list_1


def test_all_inputs_same_site_set() -> None:
    n = 4
    spins = {i: 0.5 for i in range(n)}

    km = KondoLattice2ChModel(
        spins=spins,
        hopping_0={(i, i + 1): 1.0 for i in range(n - 1)},
        hopping_1={(i + 1, i): 2.0 for i in range(n - 1)},
        u_0={i: 2.0 for i in range(n)},
        u_1={i: 3.0 for i in range(n)},
        mu_0={i: -0.1 * i for i in range(n)},
        mu_1={i: 0.2 * i for i in range(n)},
        hz_c_0={0: 0.25},
        hz_c_1={1: 0.5},
        density_density_0={(0, 1): 1.25, (1, 0): 2.5},
        density_density_1={(0, 1): 0.75, (1, 0): 1.0},
        density_density_01={(0, 1): 0.5, (1, 0): 0.25},
        hz_f={2: 0.4},
        d={3: 1.0},
        ff_exchange_xy={(0, 1): 1.25, (1, 0): 2.5},
        ff_exchange_z={(0, 1): 1.25, (1, 0): 2.5},
        kondo_xy_0={0: 1.0},
        kondo_z_0={1: 2.0},
        kondo_xy_1={2: 3.0},
        kondo_z_1={3: 4.0},
    )

    assert km.num_sites == n
    assert km.spin_list == [0.5] * n
    assert km.u_list_0 == [2.0] * n
    assert km.u_list_1 == [3.0] * n
    assert km.mu_list_0 == pytest.approx([0.0, -0.1, -0.2, -0.3])
    assert km.mu_list_1 == pytest.approx([0.0, 0.2, 0.4, 0.6])
    assert km.hz_c_list_0 == [0.25, 0.0, 0.0, 0.0]
    assert km.hz_c_list_1 == [0.0, 0.5, 0.0, 0.0]
    assert km.hz_f_list == [0.0, 0.0, 0.4, 0.0]
    assert km.d_list == [0.0, 0.0, 0.0, 1.0]

    assert km.kondo_xy_list_0 == [1.0, 0.0, 0.0, 0.0]
    assert km.kondo_z_list_0 == [0.0, 2.0, 0.0, 0.0]
    assert km.kondo_xy_list_1 == [0.0, 0.0, 3.0, 0.0]
    assert km.kondo_z_list_1 == [0.0, 0.0, 0.0, 4.0]

    assert km.hopping_0 == {(i, i + 1): 1.0 for i in range(n - 1)}
    assert km.hopping_1 == {(i + 1, i): 2.0 for i in range(n - 1)}

    assert km.density_density_0 == {(0, 1): 3.75}
    assert km.density_density_1 == {(0, 1): 1.75}
    assert km.density_density_01 == {(0, 1): 0.75}
    assert km.ff_exchange_xy == {(0, 1): 3.75}
    assert km.ff_exchange_z == {(0, 1): 3.75}

    assert km.core_model.num_sites == km.num_sites
    assert km.core_model.two_s_list == [1] * n
    assert km.core_model.hopping_0 == km.hopping_0
    assert km.core_model.hopping_1 == km.hopping_1
    assert km.core_model.u_list_0 == km.u_list_0
    assert km.core_model.u_list_1 == km.u_list_1
    assert km.core_model.mu_list_0 == km.mu_list_0
    assert km.core_model.mu_list_1 == km.mu_list_1
    assert km.core_model.hz_c_list_0 == km.hz_c_list_0
    assert km.core_model.hz_c_list_1 == km.hz_c_list_1
    assert km.core_model.density_density_0 == km.density_density_0
    assert km.core_model.density_density_1 == km.density_density_1
    assert km.core_model.density_density_01 == km.density_density_01
    assert km.core_model.hz_f_list == km.hz_f_list
    assert km.core_model.d_list == km.d_list
    assert km.core_model.ff_exchange_xy == km.ff_exchange_xy
    assert km.core_model.ff_exchange_z == km.ff_exchange_z
    assert km.core_model.kondo_xy_list_0 == km.kondo_xy_list_0
    assert km.core_model.kondo_z_list_0 == km.kondo_z_list_0
    assert km.core_model.kondo_xy_list_1 == km.kondo_xy_list_1
    assert km.core_model.kondo_z_list_1 == km.kondo_z_list_1


def test_directed_hopping_keeps_both_directions() -> None:
    km = KondoLattice2ChModel(
        spins={0: 0.5, 1: 0.5},
        hopping_0={(0, 1): 1.0, (1, 0): 2.0},
        hopping_1={(0, 1): 3.0, (1, 0): 4.0},
    )
    assert km.num_sites == 2
    assert km.spin_list == [0.5, 0.5]
    assert km.hopping_0 == {(0, 1): 1.0, (1, 0): 2.0}
    assert km.hopping_1 == {(0, 1): 3.0, (1, 0): 4.0}

    assert km.u_list_0 == [0.0, 0.0]
    assert km.u_list_1 == [0.0, 0.0]
    assert km.mu_list_0 == [0.0, 0.0]
    assert km.mu_list_1 == [0.0, 0.0]
    assert km.hz_c_list_0 == [0.0, 0.0]
    assert km.hz_c_list_1 == [0.0, 0.0]
    assert km.hz_f_list == [0.0, 0.0]
    assert km.d_list == [0.0, 0.0]
    assert km.kondo_xy_list_0 == [0.0, 0.0]
    assert km.kondo_z_list_0 == [0.0, 0.0]
    assert km.kondo_xy_list_1 == [0.0, 0.0]
    assert km.kondo_z_list_1 == [0.0, 0.0]
    assert km.density_density_0 == {}
    assert km.density_density_1 == {}
    assert km.density_density_01 == {}
    assert km.ff_exchange_xy == {}
    assert km.ff_exchange_z == {}

    assert km.core_model.num_sites == km.num_sites
    assert km.core_model.two_s_list == [1, 1]
    assert km.core_model.hopping_0 == km.hopping_0
    assert km.core_model.hopping_1 == km.hopping_1
    assert km.core_model.u_list_0 == km.u_list_0
    assert km.core_model.u_list_1 == km.u_list_1
    assert km.core_model.mu_list_0 == km.mu_list_0
    assert km.core_model.mu_list_1 == km.mu_list_1
    assert km.core_model.hz_c_list_0 == km.hz_c_list_0
    assert km.core_model.hz_c_list_1 == km.hz_c_list_1
    assert km.core_model.density_density_0 == km.density_density_0
    assert km.core_model.density_density_1 == km.density_density_1
    assert km.core_model.density_density_01 == km.density_density_01
    assert km.core_model.hz_f_list == km.hz_f_list
    assert km.core_model.d_list == km.d_list
    assert km.core_model.ff_exchange_xy == km.ff_exchange_xy
    assert km.core_model.ff_exchange_z == km.ff_exchange_z
    assert km.core_model.kondo_xy_list_0 == km.kondo_xy_list_0
    assert km.core_model.kondo_z_list_0 == km.kondo_z_list_0
    assert km.core_model.kondo_xy_list_1 == km.kondo_xy_list_1
    assert km.core_model.kondo_z_list_1 == km.kondo_z_list_1


def test_undirected_merging_for_density_density_01() -> None:
    km = KondoLattice2ChModel(
        spins={0: 0.5, 1: 0.5},
        density_density_01={(0, 1): 1.25, (1, 0): 2.5},
    )
    assert km.num_sites == 2
    assert km.spin_list == [0.5, 0.5]
    assert km.density_density_01 == {(0, 1): 3.75}

    assert km.u_list_0 == [0.0, 0.0]
    assert km.u_list_1 == [0.0, 0.0]
    assert km.mu_list_0 == [0.0, 0.0]
    assert km.mu_list_1 == [0.0, 0.0]
    assert km.hz_c_list_0 == [0.0, 0.0]
    assert km.hz_c_list_1 == [0.0, 0.0]
    assert km.hz_f_list == [0.0, 0.0]
    assert km.d_list == [0.0, 0.0]
    assert km.kondo_xy_list_0 == [0.0, 0.0]
    assert km.kondo_z_list_0 == [0.0, 0.0]
    assert km.kondo_xy_list_1 == [0.0, 0.0]
    assert km.kondo_z_list_1 == [0.0, 0.0]
    assert km.hopping_0 == {}
    assert km.hopping_1 == {}
    assert km.density_density_0 == {}
    assert km.density_density_1 == {}
    assert km.ff_exchange_xy == {}
    assert km.ff_exchange_z == {}

    assert km.core_model.num_sites == km.num_sites
    assert km.core_model.two_s_list == [1, 1]
    assert km.core_model.hopping_0 == km.hopping_0
    assert km.core_model.hopping_1 == km.hopping_1
    assert km.core_model.u_list_0 == km.u_list_0
    assert km.core_model.u_list_1 == km.u_list_1
    assert km.core_model.mu_list_0 == km.mu_list_0
    assert km.core_model.mu_list_1 == km.mu_list_1
    assert km.core_model.hz_c_list_0 == km.hz_c_list_0
    assert km.core_model.hz_c_list_1 == km.hz_c_list_1
    assert km.core_model.density_density_0 == km.density_density_0
    assert km.core_model.density_density_1 == km.density_density_1
    assert km.core_model.density_density_01 == km.density_density_01
    assert km.core_model.hz_f_list == km.hz_f_list
    assert km.core_model.d_list == km.d_list
    assert km.core_model.ff_exchange_xy == km.ff_exchange_xy
    assert km.core_model.ff_exchange_z == km.ff_exchange_z
    assert km.core_model.kondo_xy_list_0 == km.kondo_xy_list_0
    assert km.core_model.kondo_z_list_0 == km.kondo_z_list_0
    assert km.core_model.kondo_xy_list_1 == km.kondo_xy_list_1
    assert km.core_model.kondo_z_list_1 == km.kondo_z_list_1


def test_unknown_site_in_inputs_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        KondoLattice2ChModel(
            spins={0: 0.5},
            hopping_0={(0, 1): 1.0},
        )


def test_empty_spins_raises() -> None:
    with pytest.raises(ValueError):
        KondoLattice2ChModel(
            spins={},
        )
