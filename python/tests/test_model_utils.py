from edu1sces.model.utils import (
    build_site_index,
    convert_onsite_potential_to_array,
    convert_intersite_interactions_to_indexed_dict,
)


def test_build_site_index():
    site_list = [(1, 0), (0, 1), (0, 0), (0, 1)]
    site_to_integer = build_site_index(site_list)
    assert site_to_integer == {(0, 0): 0, (0, 1): 1, (1, 0): 2}


def test_convert_onsite_potential_to_array():
    site_list = [(1, 0), (0, 1), (0, 0)]
    site_to_integer = build_site_index(site_list)

    hz = {(0, 0): 0.5}
    hz_list = convert_onsite_potential_to_array(site_to_integer, hz)

    assert hz_list == [0.5, 0.0, 0.0]


def test_convert_intersite_interactions_to_indexed_dict():
    site_list = [(1, 0), (0, 1), (0, 0)]
    site_to_integer = build_site_index(site_list)

    exchange_xy = {
        ((0, 0), (1, 0)): 1.25,
        ((1, 0), (0, 0)): 2.5,
    }

    indexed = convert_intersite_interactions_to_indexed_dict(
        site_to_integer, exchange_xy, directed=False
    )

    assert indexed == {(0, 2): 3.75}
