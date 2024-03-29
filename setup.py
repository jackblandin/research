from setuptools import setup, find_packages

setup(
    name='research',
    version='0.2.0',
    description='Utility modules used for research and/or learning',
    license='MIT',
    author='Jack Blandin',
    author_email='jackblandin@gmail.com',
    url='https://github.com/jackblandin/research',
    keywords=['dqn', 'reinforcement-learning', 'machine-learning',
                'research', 'pomdp', 'python', 'deep-learning', 'gym'],
    packages=(find_packages('research/ml') + find_packages('research/rl')),
    install_requires=['ipykernel',
                      'jupyter',
                      'jupyter_contrib_nbextensions',
                      'matplotlib',
                      'numpy>=1.15.4',
                      'pandas>=0.24.1',
                      'scikit-learn>=0.20.3',
                      'scipy',
                      'seaborn',
                      'tabulate',
                      'tqdm',
                      ],
)
