from setuptools import setup, find_packages

setup(
    name='research',
    version='0.1.4',
    description='Utility modules used for research and/or learning',
    license='MIT',
    author='Jack Blandin',
    author_email='jackblandin@gmail.com',
    url='https://github.com/jackblandin/research',
    keywords=['dqn', 'reinforcement-learning', 'machine-learning',
                'research', 'pomdp', 'python', 'deep-learning'],
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas==0.20.3',
        'scikit-learn',
    ],
)
