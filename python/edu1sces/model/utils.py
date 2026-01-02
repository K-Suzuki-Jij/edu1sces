from collections import defaultdict
from .types import Site, Bond


def build_site_index(site_list: list[Site]) -> dict[Site, int]:
    """
    Build a mapping from site labels to consecutive integer indices.

    The site labels are first sorted to ensure
    a deterministic and reproducible assignment of indices.
    Each distinct site is assigned an integer index starting from 0.

    Parameters
    ----------
    site_list : list[Site]
        List of site labels.

    Returns
    -------
    dict[Site, int]
        Mapping from site label to integer index.
    """
    site_to_integer: dict[tuple, int] = {}
    for site in sorted(site_list):
        if site not in site_to_integer:
            site_to_integer[site] = len(site_to_integer)
    return site_to_integer


def convert_onsite_potential_to_array(
    site_to_integer: dict[Site, int],
    onsite_potential: dict[Site, float],
) -> list[float]:
    """
    Convert a site-resolved onsite potential into an array indexed by
    integer site indices.

    Parameters
    ----------
    site_to_integer : dict[Site, int]
        Mapping from site labels to integer site indices.
        This mapping defines both the set of sites and their ordering.
    onsite_potential : dict[Site, float]
        Mapping from site labels to onsite potential values.
        Sites not included in this dictionary are assumed to have zero potential.

    Returns
    -------
    list[float]
        Array of onsite potentials ordered by integer site index.
    """
    potential_array = [0.0] * len(site_to_integer)
    for site, potential in onsite_potential.items():
        potential_array[site_to_integer[site]] = potential
    return potential_array


def convert_intersite_interactions_to_indexed_dict(
    site_to_integer: dict[Site, int],
    intersite_interactions: dict[Bond, float],
    directed: bool,
) -> dict[tuple[int, int], float]:
    """
    Convert site-labeled intersite interactions into an integer-indexed dictionary.

    Parameters
    ----------
    site_to_integer : dict[Site, int]
        Mapping from site labels to integer site indices.
        This mapping defines the valid set of sites and their ordering.
    intersite_interactions : dict[Bond, float]
        Mapping from a bond to an interaction value.
        A bond is represented by an ordered pair of sites.
    directed : bool
        If True, bonds are treated as directed and the order (i, j) is preserved.
        If False, bonds are treated as undirected and (i, j) and (j, i) are merged.

    Returns
    -------
    dict[tuple[int, int], float]
        Mapping from integer site index pairs to interaction values.
        For undirected bonds, reversed pairs are summed into a single entry.
    """
    indexed_interactions = defaultdict(float)
    for (site_a, site_b), value in intersite_interactions.items():
        i = site_to_integer[site_a]
        j = site_to_integer[site_b]
        if directed:
            key = (i, j)
        else:
            key = (i, j) if i <= j else (j, i)
        indexed_interactions[key] += value
    return dict(indexed_interactions)


def collect_sites_from_bonds(bonds: dict[Bond, float]) -> set[Site]:
    """
    Collect all sites that appear in bond keys.

    Parameters
    ----------
    bonds : dict[Bond, float]
        Mapping from bonds to values.

    Returns
    -------
    set[Site]
        Set of sites that appear as endpoints of bonds.
    """
    site_set: set[Site] = set()
    for a, b in bonds.keys():
        site_set.add(a)
        site_set.add(b)
    return site_set
