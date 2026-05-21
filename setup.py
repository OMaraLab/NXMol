from setuptools import setup, find_packages

setup(name='chemistry_data_structure_gravy', version='0.0.1',
      description='',
      author='Josef Holownia, Callum Macfarlane, Yao Fu',
      author_email='s4782492@student.uq.edu.au',
      url='https://github.com/OMaraLab/NXMol',
      long_description='''simple data structure to represent a molecules, including relevant parsers to common file formats''',
      # packages=['src/chemistry_data_structure'],
      packages=find_packages(
          where='src',
          include=['cds_gravy*'],
          # exclude=['chemistry_data_structure/test*', chemistry_data_structure/test*]
      ),
      package_dir={"": "src"},
      install_requires=[
          'networkx',
          'matplotlib',
          'numpy',
          'pulp'
      ]
      )
