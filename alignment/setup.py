from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

    setup(
    name='pyorbit',
    version='0.1',
    description='Alignment',
    author='Justin Baker',
    author_email='baker@math.utah.edu',
    packages=['pyorbit'],  #Package Name
    )
