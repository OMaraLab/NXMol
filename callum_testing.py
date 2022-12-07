from chemistry_data_structure.objects.molecular_entity import Molecule3D
from chemistry_data_structure.parsing.input_parsers import GAMESS_to_Molecule3D
from chemistry_data_structure.helpers.field_fitting import MoleculeFieldFitter, lsq_partial_charge_fit, _gurobi_charge_fit

# from esp_analysis.ff_output_analysis import

alkane_molids = open('/mnt/c/Users/Admin/Documents/esp_analysis/alkane_molids.txt').read().split('\n')
alkane_molids = map(int, alkane_molids)

paths = [
    f'/mnt/c/Users/Admin/Documents/esp_analysis/qm_files/{molid}/wB97X_631Gd_SMD_water.out'
    for molid in alkane_molids
]

molecules = [GAMESS_to_Molecule3D(open(path).read()) for path in paths]

m = molecules[0]
lsq_partial_charge_fit(m, 0)
print('###1 molecule raw fit###')
_gurobi_charge_fit([m], verbose=True)
print('###1 molecule with total charge###')
_gurobi_charge_fit([m], verbose=True,
                   flat_sum_constraints = {'tot_charge':
                                               {'pairs':tuple((m, atom_obj) for atom_obj in m.atom_objects),
                                                'charge': 0.0}
                                           })
print('###all molecules raw fit###')
_gurobi_charge_fit(molecules, verbose=True)
print('###all molecules total charge###')
_gurobi_charge_fit(molecules, verbose=True,
                   flat_sum_constraints={f'tot_charge{jj}':
                                             {'pairs': tuple((mm, atom_obj) for atom_obj in mm.atom_objects),
                                              'charge': 0.0}
                                         for jj, mm in enumerate(molecules)})

# xxx
#
# self = MoleculeFieldFitter(molecules)
# self.generate_matrices()
# self.load_constraints(
#     symmetry_constraints={'group1': {molecules[0]: [1, 2]}},
#     sum_constraints=[{'atoms':
#                           {molecule: list(range(molecule.num_atoms))},
#                       'value': 0.0}
#                        for molecule in molecules]
# )
# self.fit()
# self.transfer_partial_charges()
# self.round_post_hoc()

xxxx
import sys
import os



import cProfile
import pstats
from random import randint, seed
self = molecules[0]
index_type = 'index'
seed(0)
profiler = cProfile.Profile()
profiler.enable()
for i in range(10000):
    index = [randint(1,7) for i in range(10)]
    [next((d for d in self._graph._node.values() if d[index_type] == i), None)
     for i in index]

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()
print('Beginning old list version')
seed(0)
profiler = cProfile.Profile()
profiler.enable()
for i in range(10000):
    index = [randint(1,7) for i in range(10)]
    [[a for a in self._graph._node.values() if a._index[index_type]==i][0] for i in index]

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()
