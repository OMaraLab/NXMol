from math import sqrt
from chemistry_data_structure.helpers.chem import MASS
from scipy.constants import pi, c


def wavenumber_to_gromacs_fc(wavenumber, atom1, atom2, backward=False, gmx_fc=None,):
    """
    Convert wavenumber from IR spectrum to GROMACS force constant (in kg/(mol*nm^
    2)).
    """
    # reduced mass in kg/mol
    reduced_mass = (MASS[atom1] * MASS[atom2] / (MASS[atom1] + MASS[atom2])) * 1e-3
    if not backward:
        """
        angular_frequency is in s^-1; since wavenumber is in cm^-1, we need to
        convert speed of light to cm/s
        """
        angular_frequency = wavenumber * 2 * pi * c * 100
        assert atom1 in MASS and atom2 in MASS, f"Unknown atom {atom1} or {atom2}"
        """
        conversion in the following refers to converting from kg/(mol*s^2) to 
        kg(mol*nm^2)
        """
        fc_pre_conversion = angular_frequency ** 2 * reduced_mass
        conv_factor = 1e-21
        
        return fc_pre_conversion * conv_factor
    
    else:
        assert atom1 in MASS and atom2 in MASS, f"Unknown atom {atom1} or {atom2}"
        assert gmx_fc is not None, "GROMACS force constant must be provided"
        conv_factor = 1e-21
        fc_pre_conversion = gmx_fc / conv_factor
        angular_frequency = sqrt(fc_pre_conversion / reduced_mass)
        wavenumber = angular_frequency / (2 * pi * c * 100)

        # return sqrt(fc_pre_conversion / reduced_mass) / (2 * pi * c * 100)
        return wavenumber
