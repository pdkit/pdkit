.. image:: https://circleci.com/gh/pdkit/pdkit.svg?style=shield
    :target: https://circleci.com/gh/pdkit/pdkit

.. image:: https://readthedocs.org/projects/pdkit/badge/
    :target: https://pdkit.readthedocs.org

To use, simply do::

    >>> import pdkit
    >>> tp = pdkit.TremorProcessor()
    >>> data_frame = tp.load_data(filename)
    >>> tp.process(dat_frame)

