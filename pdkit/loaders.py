import logging
from utils import load_data

class GaitDataLoader:

    def load_data(self, filename, format_file='cloudupdrs'):
        '''
            This is a general load data method where the format of data to load can be passed as a parameter,

            :param str filename: The path to load data from
            :param str format_file: format of the file. Default is CloudUPDRS. Set to mpower for mpower data.
        '''
        # self.data_frame = load_data(filename, format_file)
        logging.debug("{} data --> Loaded".format(format_file))
        return load_data(filename, format_file)