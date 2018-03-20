import sys
import traceback
import numpy as np

from .processor import Processor

from scipy import interpolate, signal, fft
from pywt import wavedec


class GaitProcessor(Processor):
    """Class used extract gait features from accelerometer data
    """

    def __init__(self):
        self.freeze_time = None
        self.locomotion_freeze = None
        self.freeze_index = None


    @staticmethod
    def x_numericalIntegration(x, sampling_rate):
    #
    # Do numerical integration of x with the sampling rate SR
    # -------------------------------------------------------------------
    # Copyright 2008 Marc Bachlin, ETH Zurich, Wearable Computing Lab.
    #
    # -------------------------------------------------------------------
    # I do not trust this... would like to know where it came from...
        return 1/2 * (sum(x[1:]) / sampling_rate + sum(x[:-1]) / SR)

    @staticmethod
    def estimated_autocorrelation(x):
        """
        http://stackoverflow.com/q/14297012/190597
        http://en.wikipedia.org/wiki/Autocorrelation#Estimation
        """
        n = len(x)
        variance = x.var()
        x -= x.mean()
        
        r = np.correlate(x, x, mode = 'full')[-n:]
        result = r / (variance * (np.arange(n, 0, -1)))
        return result
    

    def detect_fog(self, sample_rate=100.0, step_size=50.0):
        """Following http://delivery.acm.org/10.1145/1660000/1658515/a11-bachlin.pdf
        """
        
        # the sampling frequency was recommended by the author of the pilot study
        self.resample_signal(sampling_frequency=100.0) 
        
        data = self.data_frame.y.values
        
        NFFT = 4 * SR #256.0
        locoBand = [0.5, 3]
        freezeBand = [3, 8]
        
        windowLength = NFFT #256

        f_res = SR / NFFT
        f_nr_LBs = int(locoBand[0] / f_res)
        # f_nr_LBs[f_nr_LBs == 0] = []
        f_nr_LBe = int(locoBand[1] / f_res)
        f_nr_FBs = int(freezeBand[0] / f_res)
        f_nr_FBe = int(freezeBand[1] / f_res)

        d = NFFT / 2

        jPos = windowLength + 1
        i = 0
        
        time = []
        sumLocoFreeze = []
        freezeIndex = []
        
        while jPos < len(data):
            
            jStart = jPos - windowLength
            time.append(jPos)

            y = data[int(jStart):int(jPos)]
            y = y - np.mean(y)

            Y = fft(y, int(NFFT))
            Pyy = abs(Y*Y) / NFFT #conjugate(Y) * Y / NFFT

            areaLocoBand = self.x_numericalIntegration( Pyy[f_nr_LBs-1 : f_nr_LBe], SR )
            areaFreezeBand = self.x_numericalIntegration( Pyy[f_nr_FBs-1 : f_nr_FBe], SR )

            sumLocoFreeze.append(areaFreezeBand + areaLocoBand)

            freezeIndex.append(areaFreezeBand / areaLocoBand)

            jPos = jPos + stepSize
            i = i + 1

        self.freeze_time = time
        self.locomotion_freeze = sumLocoFreeze
        self.freeze_index = freezeIndex

