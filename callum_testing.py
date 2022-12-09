from chemistry_data_structure.objects.molecular_entity import Molecule3D
from chemistry_data_structure.parsing.input_parsers import GAMESS_to_Molecule3D
from chemistry_data_structure.helpers.field_fitting import MoleculeFieldFitter, \
    partial_charge_fit, _gurobi_charge_fit, _gurobi_post_hoc_round, _pulp_post_hoc_round, post_hoc_charge_round, \
    _lsq_charge_fit

# from esp_analysis.ff_output_analysis import

alkane_molids = open('/mnt/c/Users/Admin/Documents/esp_analysis/alkane_molids.txt').read().split('\n')
alkane_molids = map(int, alkane_molids)

paths = [
    f'/mnt/c/Users/Admin/Documents/esp_analysis/qm_files/{molid}/wB97X_631Gd_SMD_water.out'
    for molid in alkane_molids
]

molecules = [GAMESS_to_Molecule3D(open(path).read()) for path in paths]

m = molecules[0]

### validating the charge inference model. Basic case

test_mols = [GAMESS_to_Molecule3D(open(paths[0]).read()), GAMESS_to_Molecule3D(open(paths[0]).read())]
mol1 = test_mols[0]
mol2 = test_mols[1]

flat_sym = {f'g_{jj}': tuple(((mol1, mol1.atom_objects[jj]),
                              (mol2, mol2.atom_objects[jj])))
            for jj in range(mol1.num_atoms)}

flat_sym_failtest = {key: val for key, val in list(flat_sym.items())[:-1]}

_lsq_charge_fit(molecules=[mol1],
                flat_symmetry_constraints=flat_sym,
                molecules_infer=[mol2])

# _gurobi_charge_fit(molecules, verbose=True)
partial_charge_fit(m, 0)
asd = [round(a.partial_charge, 5) for a in m.atom_objects]
print('###\n\nROUNDING\n\n###')
# _gurobi_post_hoc_round(molecules, verbose=True, round_places=3)
post_hoc_charge_round([m], verbose=True,
                      round_places=3,
                      flat_sum_constraints={'tot_charge':
                                                {'pairs': tuple((m, atom_obj) for atom_obj in m.atom_objects),
                                                 'charge': 0.0}
                                            },
                      engine='pulp')
asd2 = [a.partial_charge for a in m.atom_objects]
print(asd2, sum(asd2))
print(asd, sum(asd2))
# print('###all molecules total charge###')
# _gurobi_charge_fit(molecules, verbose=True,
#                    flat_sum_constraints={f'tot_charge{jj}':
#                                              {'pairs': tuple((mm, atom_obj) for atom_obj in mm.atom_objects),
#                                               'charge': 0.0}
#                                          for jj, mm in enumerate(molecules)})

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


# import sys
# import os


# import cProfile
# import pstats
# from random import randint, seed
# self = molecules[0]
# index_type = 'index'
# seed(0)
# profiler = cProfile.Profile()
# profiler.enable()
#
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('cumtime')
# stats.print_stats()
