.. image:: https://circleci.com/gh/pdkit/pdkit.svg?style=shield
    :target: https://circleci.com/gh/pdkit/pdkit

.. image:: https://readthedocs.org/projects/pdkit/badge/
    :target: https://pdkit.readthedocs.org

#PDKIT

Example how to use pdkit to calculate Tremor amplitude and frequency:

    >>> import pdkit
    >>> tp = pdkit.TremorProcessor()
    >>> ts = pdkit.TremorTimeSeries().load(filename)
    >>> amplitude, frequency = tp.process(ts)

    where, filename is the data path to load, by default in the cloudUPDRS format.

PDKit can also read data in the MPower format, just like:

    >>> ts = pdkit.TremorTimeSeries().load(filename, 'mpower')

    where, filename is the data path to load in mpower format.

To calculate Welch, as a robust alternative to using Fast Fourier Transform, use like:

    >>> amplitude, frequency = tp.process(ts, 'welch')

Example how to use pdkit to calculate Tremor amplitude and frequency:

    >>> import pdkit
    >>> ts = pdkit.GaitTimeSeries().load(filename)
    >>> gp = pdkit.GaitProcessor()
    >>> freeze_times, freeze_indexes, locomotion_freezes = gp.freeze_of_gait(ts)
    >>> frequency_of_peaks = gp.frequency_of_peaks(ts)
    >>> speed_of_gait = gp.speed_of_gait(ts)
    >>> step_regularity, stride_regularity, walk_symmetry = gp.walk_regularity_symmetry(ts)

    where, filename is the data path to load, by default in the cloudUPDRS format.