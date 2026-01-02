import pytest
from edu1sces.model import HubbardModel


def test_minimal_hopping_only_builds_sites_and_zero_onsite() -> None:
    hm = HubbardModel(
        hopping={(0, 1): 1.0},
    )
    assert hm.num_sites == 2
    assert hm.u_list == [0.0, 0.0]
    assert hm.mu_list == [0.0, 0.0]
    assert hm.hz_list == [0.0, 0.0]
    assert hm.hopping == {(0, 1): 1.0}
    assert hm.density_density == {}
    assert hm.exchange_xy == {}
    assert hm.exchange_z == {}

    assert hm.core_model.num_sites == hm.num_sites
    assert hm.core_model.u_list == hm.u_list
    assert hm.core_model.mu_list == hm.mu_list
    assert hm.core_model.hz_list == hm.hz_list
    assert hm.core_model.hopping == hm.hopping
    assert hm.core_model.density_density == hm.density_density
    assert hm.core_model.exchange_xy == hm.exchange_xy
    assert hm.core_model.exchange_z == hm.exchange_z


def test_site_union_from_all_inputs() -> None:
    hm = HubbardModel(
        u={2: 3.0},
        mu={0: -1.0},
        hz={1: 0.25},
        hopping={(1, 2): 1.5},
    )
    assert hm.num_sites == 3
    assert hm.u_list == [0.0, 0.0, 3.0]
    assert hm.mu_list == [-1.0, 0.0, 0.0]
    assert hm.hz_list == [0.0, 0.25, 0.0]
    assert hm.hopping == {(1, 2): 1.5}
    assert hm.density_density == {}
    assert hm.exchange_xy == {}
    assert hm.exchange_z == {}

    assert hm.core_model.num_sites == hm.num_sites
    assert hm.core_model.u_list == hm.u_list
    assert hm.core_model.mu_list == hm.mu_list
    assert hm.core_model.hz_list == hm.hz_list
    assert hm.core_model.hopping == hm.hopping
    assert hm.core_model.density_density == hm.density_density
    assert hm.core_model.exchange_xy == hm.exchange_xy
    assert hm.core_model.exchange_z == hm.exchange_z


def test_directed_hopping_preserves_both_directions() -> None:
    hm = HubbardModel(
        hopping={(0, 1): 1.0, (1, 0): 2.0},
    )
    assert hm.num_sites == 2
    assert hm.u_list == [0.0, 0.0]
    assert hm.mu_list == [0.0, 0.0]
    assert hm.hz_list == [0.0, 0.0]
    assert hm.hopping == {(0, 1): 1.0, (1, 0): 2.0}
    assert hm.density_density == {}
    assert hm.exchange_xy == {}
    assert hm.exchange_z == {}

    assert hm.core_model.num_sites == hm.num_sites
    assert hm.core_model.u_list == hm.u_list
    assert hm.core_model.mu_list == hm.mu_list
    assert hm.core_model.hz_list == hm.hz_list
    assert hm.core_model.hopping == hm.hopping
    assert hm.core_model.density_density == hm.density_density
    assert hm.core_model.exchange_xy == hm.exchange_xy
    assert hm.core_model.exchange_z == hm.exchange_z


def test_undirected_density_density_merges_and_sums() -> None:
    hm = HubbardModel(
        density_density={(0, 1): 1.25, (1, 0): 2.5},
    )
    assert hm.num_sites == 2
    assert hm.u_list == [0.0, 0.0]
    assert hm.mu_list == [0.0, 0.0]
    assert hm.hz_list == [0.0, 0.0]
    assert hm.hopping == {}
    assert hm.density_density == {(0, 1): 3.75}
    assert hm.exchange_xy == {}
    assert hm.exchange_z == {}

    assert hm.core_model.num_sites == hm.num_sites
    assert hm.core_model.u_list == hm.u_list
    assert hm.core_model.mu_list == hm.mu_list
    assert hm.core_model.hz_list == hm.hz_list
    assert hm.core_model.hopping == hm.hopping
    assert hm.core_model.density_density == hm.density_density
    assert hm.core_model.exchange_xy == hm.exchange_xy
    assert hm.core_model.exchange_z == hm.exchange_z


def test_exchange_xy_is_undirected_and_merged() -> None:
    hm = HubbardModel(
        exchange_xy={(0, 1): 1.25, (1, 0): 2.5},
    )
    assert hm.num_sites == 2
    assert hm.u_list == [0.0, 0.0]
    assert hm.mu_list == [0.0, 0.0]
    assert hm.hz_list == [0.0, 0.0]
    assert hm.hopping == {}
    assert hm.density_density == {}
    assert hm.exchange_xy == {(0, 1): 3.75}
    assert hm.exchange_z == {}

    assert hm.core_model.num_sites == hm.num_sites
    assert hm.core_model.u_list == hm.u_list
    assert hm.core_model.mu_list == hm.mu_list
    assert hm.core_model.hz_list == hm.hz_list
    assert hm.core_model.hopping == hm.hopping
    assert hm.core_model.density_density == hm.density_density
    assert hm.core_model.exchange_xy == hm.exchange_xy
    assert hm.core_model.exchange_z == hm.exchange_z


def test_undirected_exchange_z_merges_and_sums() -> None:
    hm = HubbardModel(
        exchange_z={(0, 1): 1.25, (1, 0): 2.5},
    )
    assert hm.num_sites == 2
    assert hm.u_list == [0.0, 0.0]
    assert hm.mu_list == [0.0, 0.0]
    assert hm.hz_list == [0.0, 0.0]
    assert hm.hopping == {}
    assert hm.density_density == {}
    assert hm.exchange_xy == {}
    assert hm.exchange_z == {(0, 1): 3.75}

    assert hm.core_model.num_sites == hm.num_sites
    assert hm.core_model.u_list == hm.u_list
    assert hm.core_model.mu_list == hm.mu_list
    assert hm.core_model.hz_list == hm.hz_list
    assert hm.core_model.hopping == hm.hopping
    assert hm.core_model.density_density == hm.density_density
    assert hm.core_model.exchange_xy == hm.exchange_xy
    assert hm.core_model.exchange_z == hm.exchange_z


def test_accept_tuple_sites() -> None:
    a = (0, 0)
    b = (0, 1)
    hm = HubbardModel(
        u={a: 2.0},
        hopping={(a, b): 1.0, (b, a): 4.0},
        density_density={(a, b): 1.25, (b, a): 2.5},
    )
    assert hm.num_sites == 2
    assert hm.u_list == [2.0, 0.0]
    assert hm.mu_list == [0.0, 0.0]
    assert hm.hz_list == [0.0, 0.0]
    assert hm.hopping == {(0, 1): 1.0, (1, 0): 4.0}
    assert hm.density_density == {(0, 1): 3.75}
    assert hm.exchange_xy == {}
    assert hm.exchange_z == {}

    assert hm.core_model.num_sites == hm.num_sites
    assert hm.core_model.u_list == hm.u_list
    assert hm.core_model.mu_list == hm.mu_list
    assert hm.core_model.hz_list == hm.hz_list
    assert hm.core_model.hopping == hm.hopping
    assert hm.core_model.density_density == hm.density_density
    assert hm.core_model.exchange_xy == hm.exchange_xy
    assert hm.core_model.exchange_z == hm.exchange_z


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        HubbardModel()
