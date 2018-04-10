import logging
from .utils import load_data


class TremorTimeSeries:
    def __init__(self):
        # try:
        #     self.filename = filename
        # except:
        #     logging.error("Unexpected error on Tremor Time Series init: %s", sys.exc_info()[0])

        logging.debug("TremorTimeSeries init")

    def load(self, filename, format_file='cloudupdrs'):
        '''
            This is a general load data method where the format of data to load can be passed as a parameter,

            :param str filename: The path to load data from
            :param str format_file: format of the file. Default is CloudUPDRS. Set to mpower for mpower data.
            :return dataframe: data_frame.x, data_frame.y, data_frame.z: x, y, z components of the acceleration data_frame.index is the datetime-like index
        '''
        return load_data(filename, format_file)
