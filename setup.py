from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='call-center-simulator',
    version='0.1.0',
    description='Utility modules used for research and/or learning',
    long_description=readme,
    author='Jack Blandin',
    author_email='jackblandin@gmail.com',
    url='https://github.com/jackblandin/research',
    packages=find_packages(),
    install_requires=[
        'numpy=1.15.4',
        'pandas',
        'scikit-learn=0.20.3',
    ],
)
