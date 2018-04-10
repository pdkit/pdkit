
.. image:: https://circleci.com/gh/pdkit/pdkit.svg?style=svg
    :target: https://circleci.com/gh/pdkit/pdkit

To use, simply do::

    >>> import pdkit
    >>> tp = pdkit.TremorProcessor()
    >>> data_frame = tp.load_data(filename)
    >>> tp.process(dat_frame)

