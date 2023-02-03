from typing import List, FrozenSet


def bonds_for_ring(ring: List[int]) -> List[FrozenSet[int]]:
    return [
        frozenset(atom_indices)
        for atom_indices in zip(ring, ring[1:] + (ring[0],))
    ]