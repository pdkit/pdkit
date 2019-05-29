.. image:: https://circleci.com/gh/pdkit/pdkit.svg?style=shield
    :target: https://circleci.com/gh/pdkit/pdkit

.. image:: https://readthedocs.org/projects/pdkit/badge/
    :target: https://pdkit.readthedocs.org

.. image:: https://zenodo.org/badge/124572011.svg
   :target: https://zenodo.org/badge/latestdoi/124572011
   
PDKIT
#####


INSTALL INSTRUCTIONS
********************

Regular install
===============

.. code-block:: console

    $ pip install pdkit

or

.. code-block:: console

    $ pip install git+git://github.com/pdkit/pdkit.git

For "editable" install
======================

.. code-block:: console

    $ pip install -e git://github.com/pdkit/pdkit.git#egg=pdkit

For development install
=========================

.. code-block:: console

    $ git clone https://github.com/pdkit/pdkit.git
    $ pip install -r requirements.txt
    $ pip install .

TREMOR PROCESSOR
****************

Example how to use pdkit to calculate tremor amplitude and frequency:

    >>> import pdkit
    >>> tp = pdkit.TremorProcessor()
    >>> ts = pdkit.TremorTimeSeries().load(filename)
    >>> amplitude, frequency = tp.amplitude(ts)

where, `filename` is the data path to load, by default in the cloudUPDRS format.

Pdkit can also read data in the MPower format, just like:

    >>> ts = pdkit.TremorTimeSeries().load(filename, 'mpower')

where, `filename` is the data path to load in MPower format.

To calculate Welch, as a robust alternative to using Fast Fourier Transform, use like:

    >>> amplitude, frequency = tp.amplitude(ts, 'welch')

This  class also provides a method named `extract_features <http://pdkit.readthedocs.io/en/latest/tremor.html#tremor_processor.TremorProcessor.extract_features>`_
to extract all the features available in `Tremor Processor <http://pdkit.readthedocs.io/en/latest/tremor.html>`_.

    >>> tp.extract_features(ts)

BRADYKINESIA
************

    >>> import pdkit
    >>> ts = pdkit.TremorTimeSeries().load(filename)
    >>> tp = pdkit.TremorProcessor(lower_frequency=0.0, upper_frequency=4.0)
    >>> amplitude, frequency = tp.bradykinesia(ts)

GAIT
****

Example how to use pdkit to calculate various Gait features:

    >>> import pdkit
    >>> ts = pdkit.GaitTimeSeries().load(filename)
    >>> gp = pdkit.GaitProcessor()
    >>> freeze_times, freeze_indexes, locomotion_freezes = gp.freeze_of_gait(ts)
    >>> frequency_of_peaks = gp.frequency_of_peaks(ts)
    >>> speed_of_gait = gp.speed_of_gait(ts)
    >>> step_regularity, stride_regularity, walk_symmetry = gp.walk_regularity_symmetry(ts)

where, `filename` is the data path to load, by default in the CloudUPDRS format.

FINGER TAPPING
**************

Example how to use pdkit to calculate the mean alternate distance of the finger tapping tests:

    >>> import pdkit
    >>> ts = pdkit.FingerTappingTimeSeries().load(filename)
    >>> ftp = pdkit.FingerTappingProcessor()
    >>> ftp.mean_alnt_target_distance(ts)

kinesia scores (the number of key taps)

    >>> ftp.kinesia_scores(ts)

TEST RESULT SET
****************

Pdkit can be used to extract all the features for different measurements (i.e. tremor, finger tapping) placed in a single folder. The result
is a `data frame` where the measurements are rows and the columns are the features extracted.

    >>> import pdkit
    >>> testResultSet = pdkit.TestResultSet(folderpath)
    >>> testResultSet.process()

where `folderpath` is the relative folder with the different measurements. For CloudUPDRS there are measurements in the following
folder `./tests/data`. The resulting dataframe with all the features processed is saved in testResultSet.features

We can also write the `data frame` to a output file like:

    >>> testResultSet.write_output(dataframe, name)

UPDRS
****************

Pdkit can calculate the UPDRS score for a given testResultSet.

    >>> import pdkit
    >>> updrs = pdkit.UPDRS(data_frame)

The UPDRS scores can be written to a file. You can pass the name of a `filename` and the `output_format`

    >>> updrs.write_model(filename='scores', output_format='csv')

To score a new measurement against the trained knn clusters.

    >>> updrs.score(measurement)

To read the testResultSet data from a file. See TestResultSet class for more details.

    >>> updrs = pdkit.UPDRS(data_frame_file_path=file_path_to_testResultSet_file)

Clinical UPDRS
****************

Pdkit uses the clinical data to calculates classifiers implementing the k-nearest neighbors vote.


    >>> import pdkit
    >>> clinical_UPDRS = pdkit.Clinical_UPDRS(labels_file_path, data_frame)

where the `labels_file_path` is the path to the clinical data file, `data_frame` is the result of the `testResultSet`.

To score a new measurement against the trained knn clusters.

    >>> clinical_UPDRS.predict(measurement)

To read the testResultSet data from a file. See TestResultSet class for more details.

    >>> clinical_UPDRS = pdkit.Clinical_UPDRS(labels_file_path, data_frame_file_path=file_path_to_testResultSet_file)
