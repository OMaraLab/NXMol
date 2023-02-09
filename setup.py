from setuptools import setup, find_packages

setup(name='chemistry_data_structure',
      version='0.0.1',
      description='',
      author='Josef Holownia, Callum Macfarlane',
      author_email='j.holownia@uq.net.au, callum.macfarlane@uq.edu.au',
      url='https://github.com/ATB-UQ/atbmol',
      long_description='''simple data structure to represent a molecules, including relevant parsers to common file formats''',
      # packages=['src/chemistry_data_structure'],
      packages=find_packages(
          where='src',
          include=['chemistry_data_structure*'],
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
