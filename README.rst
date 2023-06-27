roxy
----

:roxy: Regression and Optimisation with X and Y errors
:Authors: Deaglan J. Bartlett and Harry Desmond
:Homepage: https://github.com/DeaglanBartlett/roxy
:Documentation: MAKE DOCUMENTATION

.. image:: https://img.shields.io/badge/astro.CO-arXiv%32307.YYYYY-B31B1B.svg
  :target: https://arxiv.org/abs/2307.YYYYY


About
=====

Installation
============

To install roxy and its dependencies in a new virtual environment, run

.. code:: bash

        python3 -m venv roxy_env
        source roxy_env/bin/activate
        git clone git@github.com:DeaglanBartlett/roxy.git
        pip install -e roxy

If you are unable to clone the repo with the above, try the https version instead

.. code:: bash

        git clone https://github.com/DeaglanBartlett/roxy.git


Licence and Citation
====================

Users are required to cite the ``roxy`` `paper <https://arxiv.org/abs/2307.YYYYY>`_, for which the following bibtex can be used

.. code:: bibtex

  @ARTICLE{roxy,
       author = {{Desmond}, H. and {Bartlett}, D.~J.},
        title = "{Marginalised Normal Regression: unbiased curve fitting in the presence of x-errors}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2023,
        month = jul,
          eid = {arXiv:2307.YYYYY},
        pages = {arXiv:2307.YYYYY},
  archivePrefix = {arXiv},
       eprint = {2307.YYYYY},
  primaryClass = {astro-ph.CO},
          url = {https://arxiv.org/abs/2307.YYYYY},
  }

and are encourgaed to cite the ``numpyro`` papers

.. code:: bibtex

  @ARTICLE{numpyro1,
	title={Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro},
	author={Phan, Du and Pradhan, Neeraj and Jankowiak, Martin},
	journal={arXiv preprint arXiv:1912.11554},
	year={2019}
    }

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


.. code:: bibtex
    @article{numpyro1,
	title={Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro},
	author={Phan, Du and Pradhan, Neeraj and Jankowiak, Martin},
	journal={arXiv preprint arXiv:1912.11554},
	year={2019}
    }

    @ARTICLE{bingham2019pyro,
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

ADD LICENCE INFORMATION HERE

Contributors
============
Below is a list of contributors to this repository.

`Deaglan Bartlett <https://github.com/DeaglanBartlett>`_ (CNRS & Sorbonne Université, Institut d’Astrophysique de Paris and Astrophysics)

`Harry Desmond <https://github.com/harrydesmond>`_ (Institute of Cosmology & Gravitation, University of Portsmouth)

Documentation
=============

MAKE DOCUMENTATION

Acknowledgements
================
DJB is supported by the Simons Collaboration on "Learning the Universe."

HD is supported by a Royal Society University Research Fellowship (grant no. 211046).
