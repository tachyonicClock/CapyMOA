Notebooks
=========

CapyMOA documentation includes Jupyter Notebooks for tutorials, and narrative
style documentation. These notebooks are run as tests to ensure they are kept
up-to-date. This document explains how to run, render and test notebooks.

* Added to the ``/notebooks`` directory.
* Rendered to HTML and included in the documentation of the website using 
* To add a notebook to the documentation, add the notebook to the ``/notebooks``
  directory and add the filename to the ``toctree`` in ``notebooks/index.rst``.
* Please check the notebooks are being converted and included in the documentation
  by building the documentation locally. See :doc:`/contributing/docs`.
* The parser for markdown used by Jupiter Notebooks is different from the one
  used by nbsphinx. This can lead to markdown rendering unexpectedly you might
  need to adjust the markdown in the notebooks to render correctly on the website.

    * Bullet points should have a newline after the bullet point.
      
      .. code-block:: markdown

        * Bullet point 1

        * Bullet point 2

Slow Notebooks
--------------

Some notebooks may take a long time to run. The preferred way to handle this is
to add hidden cells that simplify the notebook for testing.


Hiding cells
------------

To hide a cell, add the ``remove-cell`` tag to the cell metadata. This is usually
achievable through the Jupyter Notebook interface by selecting the cell and


.. code-block:: json

    {
     "tags": [
      "remove-cell"
     ]
    }



Testing or Overwriting Notebook Output
--------------------------------------

The ``tasks.py`` defines aliases for running the notebooks as tests or for
overwriting the outputs of the notebooks. To run the notebooks as tests:

.. code-block:: bash

    invoke test.nb # add --help for options

.. program-output:: python -m invoke test.nb --help
