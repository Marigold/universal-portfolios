from setuptools import setup, find_packages

setup(name='universal-portfolios',
      version='0.1',
      description='Collection of algorithms for online portfolio selection',
      url = 'https://github.com/Marigold/universal-portfolios',
      download_url = 'https://github.com/marigold/universal-portfolios/tarball/0.1',
      author='Mojmir Vinkler',
      author_email='mojmir.vinkler@gmail.com',
      license='MIT',
      packages=find_packages(),
      package_data={
      	'universal': ['data/*.pkl']
      },
      keywords=['portfolio'],
      install_requires = ['pandas>=0.13.1', 'cvxopt>=1.1.6', 'matplotlib>=1.3.1'],
      zip_safe=False)
