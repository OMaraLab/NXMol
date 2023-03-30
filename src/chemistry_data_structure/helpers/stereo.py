"""
Helper functions for handling stereochemistry.
"""
import sys

from chemistry_data_structure.objects.molecular_entity import Molecule3D
from chemistry_data_structure.parsing.input_parsers import pdb_to_Molecule3D
import os
import re
import json
from urllib.request import urlopen, urlretrieve
from time import sleep
import subprocess

# TODO: REMOVE THIS!!
JMOL_EXEC_PATH = 'C:\\Users\\joeho\\ATB\\jmol-16.1.7\\JmolData.jar'

def getChiralCenters(mol: Molecule3D):
    """
    Use jmol to detect chiral centers in a molecule.
    :param mol: a 3D molecule object to determine the chirality of through its PDB representation.
    :return: a dictionary of {atom_name: R/S}
    """

    # adjacent strings are read as one string, backslash character after " creates new line continuation in the python code
    # 's'=='S' is true according to jmol
    # Code to load a pdb str, take the chirality and output it in dictionary format
    CODE_BLOCK = r"load data \"pdb mol\";" + \
                 '\n' + mol.pdbStr() + \
                 r"end \"pdb mol\";" \
                 r"holmium_str = 'holmium';" \
                 r"holmium_error = 'ERROR HOLMIUM_ATOM';" \
                 r"if (getproperty('atominfo[SELECT * WHERE element==holmium_str]').length != 0)" \
                 r"{write VAR holmium_error 'MOL_NAME.chi'; exitJmol;};" \
                 r"calculate chirality;" \
                 r"var x = {*}.chirality;" \
                 r"var mol_shape = {*}.shape;" \
                 r"var shape_error = 'ERROR TRIGONAL_PLANER_STEREOCENTER';" \
                 r"var atomnames = getproperty('atominfo[SELECT info]');" \
                 r"var c = '{';" \
                 r"for (var i = 1; i<=x.length;i++)" \
                 r" {" \
                 r"if (x[i] != '' & x[i] != 'Z' & x[i] != 'E' & mol_shape[i]=='trigonal planar')" \
                 r" {write VAR shape_error 'MOL_NAME.chi'; exitJmol;}" \
                 r"else if (x[i] != '' & c != '{')" \
                 r" {c = c+ ',\"' + atomnames[i]['info'] + '\":\"' + x[i] + '\"';} " \
                 r"else if (x[i] != '')" \
                 r"{c = c+ '\"' + atomnames[i]['info'] + '\":\"' + x[i] + '\"';};" \
                 r"};" \
                 r"c = c+'}';" \
                 r"print c;" \

    CODE_BLOCK = CODE_BLOCK.replace('MOL_NAME', mol.name)

    # JmolData.jar is the same as Jmol.jar but with no graphical interface, replacing JmolData.jar with Jmol.jar
    # allows you to open the console after it has run to debug -n ensures no screen display is used,
    # -o if you require jmol messages to output to the command line, -j takes the CODE_BLOCK as an argument,
    # see http://wiki.jmol.org/index.php/Jmol_Application#Command_line_options
    raw_result = subprocess.check_output(['java', '-jar', JMOL_EXEC_PATH, '-n', '-i', '-j', '"' + CODE_BLOCK + '"'], timeout=0.5 * 60)

    # extract final dictionary of chiral centers as a string from output
    filtered_result = '{' + str(raw_result).split('{')[1].split('}')[0] + '}'
    chiral_centers = {}
    for k, v in json.loads(filtered_result).items():
        print('Original: ', k)
        atom_info = re.split(r'(\d+)', k.strip())
        print('Split List: ', atom_info)
        atom_name = atom_info[0][0] + atom_info[1]  # grab element and number to make atom name
        chiral_centers[atom_name] = v

    print("Chiral Centers: ", chiral_centers)
    return chiral_centers


if __name__ == "__main__":

    # DEBUG ONLY
    fname = "../test/data/pdb/alanine.pdb"
    #fname = "../test/data/pdb/esketamine.pdb"
    mol_name = "esketamine"
    with open(fname, 'r') as pdb_file:
        pdb_str = pdb_file.read()

    mol = pdb_to_Molecule3D(pdb_str, mol_name=mol_name, net_charge=0, assign_bond_orders_and_charges=True)
    getChiralCenters(mol)