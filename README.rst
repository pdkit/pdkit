.. image:: https://img.shields.io/badge/license-MIT-yellowgreen
    :target: https://github.com/pdkit/pdkit/blob/master/LICENSE
    
.. image:: https://img.shields.io/badge/release-1.4.3-blue
    :target: https://pypi.org/project/pdkit/
    
.. image:: https://circleci.com/gh/pdkit/pdkit.svg?style=shield
     :target: https://circleci.com/gh/pdkit/pdkit

.. image:: https://zenodo.org/badge/124572011.svg
   :target: https://zenodo.org/badge/latestdoi/124572011
   
PDkit
#####

PDkit is a python module that provides a comprehensive toolkit for the management and processing of Parkinson's symptoms performance data captured by high-use-frequency smartphone apps and continuously by wearables. PDkit facilitates the application of an extensive collection of methods and techniques across all stages of the Parkinson's information processing pipeline. Although inherently flexible, PDkit currently prioritises functionalities critical to therapeutic clinical trial delivery rather than general patient care.

More information is available in the following paper:

Stamate C, Saez Pons J, Weston D, Roussos G (2021) PDKit: A data science toolkit for the digital assessment of Parkinson’s Disease. PLoS Comput Biol 17(3): e1008833. doi:10.1371/journal.pcbi.1008833 https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008833

An example of how PDkit is used in clinical strudies of Parkinson's can be found in:

Jha, A., Menozzi, E., Oyekan, R. et al. The CloudUPDRS smartphone software in Parkinson’s study: cross-validation against blinded human raters. npj Parkinsons Dis. 6, 36 (2020). https://doi.org/10.1038/s41531-020-00135-w

The PDkit currently supports directly the following apps:  `cloudUPDRS <http://www.updrs.net>`_, `mPower <https://parkinsonmpower.org/>`_, `HopkinsPD <https://github.com/zad/HopkinsPD_Android>`_ and `OPDC <https://www.opdc.ox.ac.uk/opdc-smartphone-app-tests-for-early-signs-of-parkinson-s>`_.

Full documentation of PDkit features is available on `readthedocs <http://pdkit.readthedocs.io/en/latest/>`_.

Installation instructions and brief use examples are included below. Additional examples of PDkit use can be found in the *notebooks* directory including data file samples to help explore how the toolkit can be used.

Installation Instructions
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

How to Use
************************

Tremor
=========================

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

Bradykinesia
=========================

    >>> import pdkit
    >>> ts = pdkit.TremorTimeSeries().load(filename)
    >>> tp = pdkit.TremorProcessor(lower_frequency=0.0, upper_frequency=4.0)
    >>> amplitude, frequency = tp.bradykinesia(ts)

Gait
=========================

Example how to use pdkit to calculate various Gait features:

    >>> import pdkit
    >>> ts = pdkit.GaitTimeSeries().load(filename)
    >>> gp = pdkit.GaitProcessor()
    >>> freeze_times, freeze_indexes, locomotion_freezes = gp.freeze_of_gait(ts)
    >>> frequency_of_peaks = gp.frequency_of_peaks(ts)
    >>> speed_of_gait = gp.speed_of_gait(ts)
    >>> step_regularity, stride_regularity, walk_symmetry = gp.walk_regularity_symmetry(ts)

where, `filename` is the data path to load, by default in the CloudUPDRS format.

Finger Tapping
=========================

Example how to use pdkit to calculate the mean alternate distance of the finger tapping tests:

    >>> import pdkit
    >>> ts = pdkit.FingerTappingTimeSeries().load(filename)
    >>> ftp = pdkit.FingerTappingProcessor()
    >>> ftp.mean_alnt_target_distance(ts)

kinesia scores (the number of key taps)

    >>> ftp.kinesia_scores(ts)

Process a full data set
=========================

Pdkit can be used to extract all the features for different measurements (i.e. tremor, finger tapping) placed in a single folder. The result
is a `data frame` where the measurements are rows and the columns are the features extracted.

    >>> import pdkit
    >>> testResultSet = pdkit.TestResultSet(folderpath)
    >>> testResultSet.process()

where `folderpath` is the relative folder with the different measurements. For CloudUPDRS there are measurements in the following
folder `./tests/data`. The resulting dataframe with all the features processed is saved in testResultSet.features

We can also write the `data frame` to a output file like:

    >>> testResultSet.write_output(dataframe, name)

Learn UPDRS scores from data
============================

Pdkit can calculate the UPDRS score for a given testResultSet.

    >>> import pdkit
    >>> updrs = pdkit.UPDRS(data_frame)

The UPDRS scores can be written to a file. You can pass the name of a `filename` and the `output_format`

    >>> updrs.write_model(filename='scores', output_format='csv')

To score a new measurement against the trained knn clusters.

    >>> updrs.score(measurement)

To read the testResultSet data from a file. See TestResultSet class for more details.

    >>> updrs = pdkit.UPDRS(data_frame_file_path=file_path_to_testResultSet_file)

Learn UPDRS from clinical scores
========================================

Pdkit uses the clinical data to calculates classifiers implementing the k-nearest neighbors vote.


    >>> import pdkit
    >>> clinical_UPDRS = pdkit.Clinical_UPDRS(labels_file_path, data_frame)

where the `labels_file_path` is the path to the clinical data file, `data_frame` is the result of the `testResultSet`.

To score a new measurement against the trained knn clusters.

    >>> clinical_UPDRS.predict(measurement)

To read the testResultSet data from a file. See TestResultSet class for more details.

    >>> clinical_UPDRS = pdkit.Clinical_UPDRS(labels_file_path, data_frame_file_path=file_path_to_testResultSet_file)
