roxy
----

:roxy: Regression and Optimisation with X and Y errors
:Authors: Deaglan J. Bartlett and Harry Desmond
:Homepage: https://github.com/DeaglanBartlett/roxy
:Documentation: https://roxy.readthedocs.io/en/latest/

.. image:: https://readthedocs.org/projects/roxy/badge/?version=latest
  :target: https://roxy.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://img.shields.io/badge/astro.CO-arXiv%32309.00948-B31B1B.svg
  :target: https://arxiv.org/abs/2309.00948

.. image:: https://github.com/DeaglanBartlett/roxy/actions/workflows/python-app.yml/badge.svg
  :target: https://github.com/DeaglanBartlett/roxy/actions/workflows/python-app.yml
  :alt: Build Status


About
=====

``roxy`` (Regression and Optimisation with X and Y errors) is a python package for performing
MCMC where the data have both x and y errors. The common approach for this problem is to use a
Gaussian likelihood with a mean given by :math:`f(x_{\rm obs}, \theta)` and a variance
:math:`\sigma_y^2 + f^\prime(x_{\rm obs}, \theta)^2 \sigma_x^2`, but this ignores the underlying
distribution of true x values and thus gives biased results. Instead, this package allows
one to use the MNR (Marginalised Normal Regression) method which does not exhibit such 
biases. 

The code uses automatic differentiation enabled by jax to both sample the
likelihood using Hamiltonian Monte Carlo and to compute the derivatives 
required for the likelihood. We employ the NUTS method implemented in ``numpyro``
for fast sampling. For the galaxy cluster example in the MNR paper 
(which contains over 250 data points), a single chain run on a laptop performs 
approximately 3500 iterations per second, such that a chain with 700 warm-up
steps and 5000 samples takes approximately 1.6 seconds to sample, and gives
over 3500 effective samples for each of the parameters, with Gelman Rubin statistics 
equal to unity within less than 0.01. Given its efficiency and simplicity to use (one 
need to just define the function of interest, the parameters to sample and their
prior ranges), we advocate for its use not just in the presence of x errors,
but also without these.

As well as returning posterior samples and allowing likelihood computations
(which can be integrated into the user's larger code), ``roxy`` is interfaced with 
``arviz`` to produce trace plots, ``corner`` and ``getdist`` to make two-dimensional
posterior plots, and ``fgivenx`` for posterior predictive plots. See below for 
the relevant citations one must use if one uses these modules.


Installation
============

Requirements
^^^^^^^^^^^^

Since ``roxy`` is a python package, the user will need python3 installed.
We have tested ``roxy`` using python3.11, so suggest that the user also uses
this python version.

If one wants to reproduce the results of the ``roxy`` `paper <https://arxiv.org/abs/2309.00948>`_,
one needs to run the example given in ``roxy.examples.bias_example.py``, which requires
``mpi4py``. 
This is not installed by default (see below), but if you do want to install it, you will need
mpi to be installed beforehand. This can be done by running the following

MacOS:

.. code:: bash

	brew install open-mpi 

Ubuntu:

.. code:: bash

	sudo apt-get install openmpi-bin libopenmpi-dev




Installing
^^^^^^^^^^

To install roxy and its dependencies in a new virtual environment, run

.. code:: bash

        python3 -m venv roxy_env
        source roxy_env/bin/activate
        git clone git@github.com:DeaglanBartlett/roxy.git
        pip install -e roxy

If you are unable to clone the repo with the above, try the https version instead

.. code:: bash

        git clone https://github.com/DeaglanBartlett/roxy.git

To run the script ``roxy.examples.bias_example.py``, you will need to install ``mpi4py``
which can be done alongside installing ``roxy`` by, instead of using the ``pip install``
instruction above,  running

.. code:: bash

	pip install -e "roxy[all]"


Licence and Citation
====================

Users are required to cite the ``roxy`` `paper <https://arxiv.org/abs/2309.00948>`_, for which the following bibtex can be used

.. code:: bibtex

  @ARTICLE{roxy,
       author = {{Bartlett}, D.~J. and {Desmond}, H.},
        title = "{Marginalised Normal Regression: unbiased curve fitting in the presence of x-errors}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2023,
        month = sep,
          eid = {arXiv:2309.00948},
        pages = {arXiv:2309.00948},
	  doi = {10.48550/arXiv.2309.00948},
  archivePrefix = {arXiv},
       eprint = {2309.00948},
  primaryClass = {astro-ph.CO},
          url = {https://arxiv.org/abs/2309.00948},
  }

and are encourgaed to cite the ``numpyro`` papers

.. code:: bibtex

  @ARTICLE{numpyro1,
	title={Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro},
	author={Phan, Du and Pradhan, Neeraj and Jankowiak, Martin},
	journal={arXiv preprint arXiv:1912.11554},
	year={2019}
    }


.. code:: bibtex

  @ARTICLE{numpyro2,
	author    = {Eli Bingham and
	       Jonathan P. Chen and
	       Martin Jankowiak and
	       Fritz Obermeyer and
	       Neeraj Pradhan and
	       Theofanis Karaletsos and
	       Rohit Singh and
	       Paul A. Szerlip and
	       Paul Horsfall and
	       Noah D. Goodman},
	title     = {Pyro: Deep Universal Probabilistic Programming},
	journal   = {J. Mach. Learn. Res.},
	volume    = {20},
	pages     = {28:1--28:6},
	year      = {2019},
	url       = {http://jmlr.org/papers/v20/18-403.html}
    }

Additionally, if you use the function ``roxy.plotting.posterior_predictive_plot``, then, as this used the ``fgivenx`` `package <https://fgivenx.readthedocs.io/en/latest/?badge=latest>`_, you must cite

.. code:: bibtex

   @article{fgivenx,
       doi = {10.21105/joss.00849},
       url = {http://dx.doi.org/10.21105/joss.00849},
       year  = {2018},
       month = {Aug},
       publisher = {The Open Journal},
       volume = {3},
       number = {28},
       author = {Will Handley},
       title = {fgivenx: Functional Posterior Plotter},
       journal = {The Journal of Open Source Software}
   }


We also provide simple routines to plot posterior distribtuions with ``roxy.plotting.triangle_plot``. If you use ``module="corner"`` with this function, please cite

.. code:: bibtex

   @article{corner,
	doi = {10.21105/joss.00024},
	url = {https://doi.org/10.21105/joss.00024},
	year  = {2016},
	month = {jun},
	publisher = {The Open Journal},
	volume = {1},
	number = {2},
	pages = {24},
	author = {Daniel Foreman-Mackey},
	title = {corner.py: Scatterplot matrices in Python},
	journal = {The Journal of Open Source Software}
    }

and if you use ``module="getdist"``, please cite

.. code:: bibtex

   @article{getdist,
      author         = "Lewis, Antony",
      title          = "{GetDist: a Python package for analysing Monte Carlo
                        samples}",
      year           = "2019",
      eprint         = "1910.13970",
      archivePrefix  = "arXiv",
      primaryClass   = "astro-ph.IM",
      SLACcitation   = "%%CITATION = ARXIV:1910.13970;%%",
      url            = "https://getdist.readthedocs.io"
     }

MIT License

Copyright (c) 2023 Deaglan John Bartlett

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Contributors
============
Below is a list of contributors to this repository.

`Deaglan Bartlett <https://github.com/DeaglanBartlett>`_ (CNRS & Sorbonne Université, Institut d’Astrophysique de Paris and Astrophysics)

`Harry Desmond <https://github.com/harrydesmond>`_ (Institute of Cosmology & Gravitation, University of Portsmouth)

Documentation
=============

The documentation for this project can be found
`at this link <https://roxy.readthedocs.io/>`_

Acknowledgements
================
DJB is supported by the Simons Collaboration on "Learning the Universe."

HD is supported by a Royal Society University Research Fellowship (grant no. 211046).
