import sys
import traceback
import numpy as np
from .tremor_processor import TremorProcessor

from scipy import interpolate, signal, fft
import matplotlib.pyplot as plt

SR = 100.0            # Sample rate in herz
stepSize = SR/2      # Step size in samples
offDelay = 2.0       # Evaluation delay in seconds: tolerates delay after detecting
onDelay=2.0          # Evaluation delay in seconds: tolerates delay before detecting

NANOSEC_TO_SEC = 1000000000.0
MILLISEC_TO_SEC = 1000.0
SAMPLING_FREQUENCY = 100.0 # this was recommended by the author of the pilot study [1]
CUTOFF_FREQUENCY_HIGH = 2.0 # Hz as per [1]
FILTER_ORDER = 2 # as per [1]
WINDOW = 256 # this was recommended by the author of the pilot study [1]
LOWER_FREQUENCY_TREMOR = 2.0 # Hz as per [1]
UPPER_FREQUENCY_TREMOR = 10.0 # Hz as per [1]
CUTOFF_FREQUENCY_LOW = 4.0 # Hz as per [1]z

def x_numericalIntegration(x, SR):
#
# Do numerical integration of x with the sampling rate SR
# -------------------------------------------------------------------
# Copyright 2008 Marc Bachlin, ETH Zurich, Wearable Computing Lab.
#
# -------------------------------------------------------------------
# I do not trust this... would like to know where it came from...
    return 1/2 * (sum(x[1:]) / SR + sum(x[:-1]) / SR)

# inheriting from TremorProcessor to stub loading the data
# will fix this hack once the DataLoader is implemented
class GaitProcessor(TremorProcessor):
    """Class used extract gait features from accelerometer data
    """
    
    def detect_fog(self, sample_rate=100.0, step_size=50.0, plot=False):
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

            areaLocoBand = x_numericalIntegration( Pyy[f_nr_LBs-1 : f_nr_LBe], SR )
            areaFreezeBand = x_numericalIntegration( Pyy[f_nr_FBs-1 : f_nr_FBe], SR )

            sumLocoFreeze.append(areaFreezeBand + areaLocoBand)

            freezeIndex.append(areaFreezeBand / areaLocoBand)

            jPos = jPos + stepSize
            i = i + 1

        self.freeze_time = time
        self.locomotion_freeze = sumLocoFreeze
        self.freeze_index = freezeIndex

        if plot:
            plt.plot(self.freeze_time, self.freeze_index)
            plt.plot(self.freeze_time, self.locomotion_freeze)
            plt.show()