from setuptools import setup

setup(name='universal',
      version='0.1',
      description='Collection of algorithms for online portfolio selection',
      url='http://github.com/storborg/funniest',
      author='Mojmir Vinkler',
      author_email='mojmir.vinkler@gmail.com',
      license='MIT',
      packages=['universal'],
      keywords=['portfolio'],
      install_requires = ['pandas>=0.13.1', 'cvxopt>=1.1.6', 'matplotlib>=1.3.1'],
      zip_safe=False)
