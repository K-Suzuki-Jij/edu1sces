import pytest
from edu1sces.model import HeisenbergModel


def test_minimal_exchange_xy_only() -> None:
    n = 4
    hm = HeisenbergModel(
        spins={i: 0.5 for i in range(n)},
        exchange_xy={(i, i + 1): 1.0 for i in range(n - 1)},
    )

    assert hm.site_to_integer == {0: 0, 1: 1, 2: 2, 3: 3}
    assert hm.num_sites == n
    assert hm.spin_list == [0.5] * n
    assert hm.hz_list == [0.0] * n
    assert hm.d_list == [0.0] * n
    assert hm.exchange_xy == {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0}
    assert hm.exchange_z == {}

    assert hm.core_model.num_sites == hm.num_sites
    assert hm.core_model.two_s_list == [1] * n
    assert hm.core_model.hz_list == hm.hz_list
    assert hm.core_model.d_list == hm.d_list
    assert hm.core_model.exchange_xy == hm.exchange_xy
    assert hm.core_model.exchange_z == hm.exchange_z


def test_accept_tuple_sites() -> None:
    hm = HeisenbergModel(
        spins={(0, 0): 0.5, (0, 1): 0.5},
        exchange_xy={((0, 0), (0, 1)): 1.0},
        exchange_z={((0, 0), (0, 1)): 2.0},
        hz={(0, 0): 0.1},
        d={(0, 1): 0.2},
    )

    assert hm.site_to_integer == {(0, 0): 0, (0, 1): 1}
    assert hm.num_sites == 2
    assert hm.spin_list == [0.5, 0.5]
    assert hm.hz_list == [0.1, 0.0]
    assert hm.d_list == [0.0, 0.2]
    assert hm.exchange_xy == {(0, 1): 1.0}
    assert hm.exchange_z == {(0, 1): 2.0}

    assert hm.core_model.num_sites == hm.num_sites
    assert hm.core_model.two_s_list == [1, 1]
    assert hm.core_model.hz_list == hm.hz_list
    assert hm.core_model.d_list == hm.d_list
    assert hm.core_model.exchange_xy == hm.exchange_xy
    assert hm.core_model.exchange_z == hm.exchange_z


def test_missing_spin_for_exchange_site_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        HeisenbergModel(
            spins={0: 0.5},
            exchange_xy={(0, 1): 1.0},
        )


def test_bond_key_order_is_merged_and_summed() -> None:
    n = 3
    hm = HeisenbergModel(
        spins={i: 0.5 for i in range(n)},
        exchange_xy={(0, 1): 1.0, (1, 0): 2.0},
    )

    assert hm.site_to_integer == {0: 0, 1: 1, 2: 2}
    assert hm.num_sites == n
    assert hm.spin_list == [0.5] * n
    assert hm.hz_list == [0.0] * n
    assert hm.d_list == [0.0] * n
    assert hm.exchange_xy == {(0, 1): 3.0}
    assert hm.exchange_z == {}

    assert hm.core_model.num_sites == hm.num_sites
    assert hm.core_model.two_s_list == [1] * n
    assert hm.core_model.hz_list == hm.hz_list
    assert hm.core_model.d_list == hm.d_list
    assert hm.core_model.exchange_xy == hm.exchange_xy
    assert hm.core_model.exchange_z == hm.exchange_z
