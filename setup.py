from setuptools import setup

setup(
    name='roxy',
    version='0.1.0',
    description='A Python package to perform Regression and Optimisation with X and Y errors',
    url='https://github.com/DeaglanBartlett/roxy',
    author='Deaglan Bartlett and Harry Desmond',
    author_email='deaglan.bartlett@physics.ox.ac.uk',
    license='MIT licence',
    packages=['roxy'],
    install_requires=[
        'numpy',
        'jax',
        'jaxlib',
        'scipy',
        'numpyro',
        'matplotlib',
	'corner',
	'getdist',
	'arviz',
	'fgivenx',
        ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)
