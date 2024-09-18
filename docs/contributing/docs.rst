Adding Documentation
====================

Prerequisites
-------------

#.  Ensure that you have `Pandoc <https://pandoc.org/>`_ installed on your system.
    If it's not installed, you can install it by running the following command on

    .. tab-set::

        .. tab-item:: Ubuntu

            .. code-block:: bash

                sudo apt-get install -y pandoc

        .. tab-item:: macOS

            .. code-block:: bash

                sudo brew install pandoc

        .. tab-item:: Window/Other

            Follow the instructions on the `Pandoc website <https://pandoc.org/installing.html>`_.

        .. tab-item:: conda

            .. code-block:: bash

                conda install -c conda-forge pandoc

#.  Install the documentation dependencies by running the following command in
    the root directory of the repository (make sure you are in the correct
    virtual/conda environment):

    .. code-block:: bash

        pip install --editable ".[dev,doc]"

Building Documentation
----------------------

To build the documentation, run the following command in the project root:

.. code-block:: bash

    python -m invoke docs.build

.. program-output:: python -m invoke docs.build --help

Once built, you can visit the documentation locally in your browser.

.. note::

    If you run into nitpicky errors, you can allow a more permissive documentation
    build with:

    .. code-block:: bash

        python -m invoke docs.build -i

    Continuous integration will still run the strict build, so make sure to fix
    any errors before making a pull request.


Types of Documentation
----------------------

Auto-Generated Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Auto-generated documentation from the source code using Autodoc. See this
[link](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
for more information. CapyMOA, for now is using Sphinx/reStructuredText style
docstrings. Rather than having type information in the docstring, we prefer to
use Python-type hints.


.. code-block:: python

    class Stream:
        """A datastream that can be learnt instance by instance."""

        def __init__(
            self,
            moa_stream: Optional[InstanceStream] = None,
            schema: Optional[Schema] = None,
            CLI: Optional[str] = None,
        ):
            """Construct a Stream from a MOA stream object.

            Usually, you will want to construct a Stream using the :func:`stream_from_file`
            function.

            :param moa_stream: The MOA stream object to read instances from. Is None
                if the stream is created from a numpy array.
            :param schema: The schema of the stream. If None, the schema is inferred
                from the moa_stream.
            :param CLI: Additional command line arguments to pass to the MOA stream.
            :raises ValueError: If no schema is provided and no moa_stream is provided.
            :raises ValueError: If command line arguments are provided without a moa_stream.
            """

Jupyter Notebooks
~~~~~~~~~~~~~~~~~

Parts of CapyMOA's documentation is written as Jupyter Notebooks. See
:doc:`/contributing/notebooks`.


Manual Documentation
~~~~~~~~~~~~~~~~~~~~

Manually written documentation in the ``/docs`` directory. These can be written in
reStructuredText or Markdown. To add a new page to the documentation, add a new
file to the ``/docs`` directory and add the filename to the ``toctree`` in ``index.rst``
or the appropriate location in the documentation.
