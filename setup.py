from setuptools import setup, find_packages

setup(
    name='SCHull',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'torch',
    ],
    author='Shih-Hsin Wang',
    description='A Sparse, Connected, and Rigid Graph for 3D point clouds',
    url='https://github.com/shihhsinwang0214/SCHull',
    python_requires='>=3.6',
)
