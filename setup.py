from setuptools import setup, find_packages

import os 
current_folder = os.path.dirname(os.path.abspath(__file__))
version = '0.0.0.0.0.0' # year.month.day.hour.minute.second
with open(os.path.join(current_folder,'VERSION')) as version_file:
    version = version_file.read().strip()

setup(name='gvnr',
      version=version,
      description='gvnr: python package for the paper: "Global Vectors for Node Representation"',
      url='https://github.com/brochier/gvnr',
      author='Robin Brochier',
      author_email='robin.brochier@univ-lyon2.fr',
      license='MIT',
      include_package_data=True,
      packages=find_packages(exclude=['docs', 'tests*']),
      package_data={'': ['gvnr/resources/*', 'gvnr/conf.yml']},
      install_requires=[
          'numpy',
          'scipy',
          'sklearn',
          'tensorflow',
          'theano',
          'matplotlib',
          'gensim'
      ],
      zip_safe=False)
