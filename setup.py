from distutils.core import setup
from setuptools import find_packages


def parse_requirements(filename):
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


install_reqs = parse_requirements('requirements.txt')

setup(name='universal-portfolios',
      version='0.3.3',
      description='Collection of algorithms for online portfolio selection',
      url='https://github.com/Marigold/universal-portfolios',
      download_url='https://github.com/Marigold/universal-portfolios/archive/master.zip',
      author='Mojmir Vinkler',
      author_email='mojmir.vinkler@gmail.com',
      license='MIT',
      packages=find_packages(),
      package_data={
          'universal': ['data/*.pkl']
      },
      keywords=['portfolio'],
      install_requires=install_reqs,
      zip_safe=False)
