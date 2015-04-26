from distutils.core import setup
from setuptools import find_packages
from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]

setup(name='universal-portfolios',
      version='0.1',
      description='Collection of algorithms for online portfolio selection',
      url='https://github.com/Marigold/universal-portfolios',
      download_url='https://github.com/marigold/universal-portfolios/tarball/0.1',
      author='Mojmir Vinkler',
      author_email='mojmir.vinkler@gmail.com',
      license='MIT',
      packages=find_packages(),
      package_data={
            'universal': ['data/*.pkl']
      },
      keywords=['portfolio'],
      install_requires=reqs,
      zip_safe=False)
