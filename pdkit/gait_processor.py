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

    def get_frequency_from_peaks(self, start_offset=100, end_offset=100, delta=0.5):
        # this method calculatess the frequency from the peaks of the x-axis acceleration
        self.peaks_data_frame = self.data_frame[start_offset:-end_offset]

        maxtab, mintab = np.peakdet(self.peaks_data_frame.x, delta)

        x = np.mean(self.peaks_data_frame.dt[maxtab[1:,0].astype(int)] - self.peaks_data_frame.dt[maxtab[0:-1,0].astype(int)])
        
        self.frequency_from_peaks = 1/x

    def calc_gait_speed(self, wavelet_type='db3', wavelet_level=6):
        # the technique followed in this method is described in detail in [2]
        # it involves wavelet transforming the signal and calculating
        # the gait speed from the energies of the approximation coefficients
        coeffs = wavedec(self.data_frame.mag_sum_acc, wavelet=wavelet_type, level=wavelet_level)

        energy = [sum(coeffs[wavelet_level - i]**2) / len(coeffs[wavelet_level - i]) for i in range(wavelet_level)]

        WEd1 = energy[0] / (5 * np.sqrt(2))
        WEd2 = energy[1] / (4 * np.sqrt(2))
        WEd3 = energy[2] / (3 * np.sqrt(2))
        WEd4 = energy[3] / (2 * np.sqrt(2))
        WEd5 = energy[4] / np.sqrt(2)
        WEd6 = energy[5] / np.sqrt(2)

        speed= 0.5 * np.sqrt(WEd1+(WEd2/2)+(WEd3/3)+(WEd4/4)+(WEd5/5))

        self.gait_speed = speed

    def calc_regularity_symmetry(self):
        
        def regularity_symmetry(v):
            maxtab, _ = np.peakdet(v, DELTA)
            return maxtab[1][1], maxtab[2][1]

        step_regularity_x, stride_regularity_x = regularity_symmetry(self.estimated_autocorrelation(self.data_frame.x))
        step_regularity_y, stride_regularity_y = regularity_symmetry(self.estimated_autocorrelation(self.data_frame.x))
        step_regularity_z, stride_regularity_z = regularity_symmetry(self.estimated_autocorrelation(self.data_frame.x))

        self.step_regularity_x = step_regularity_x
        self.stride_regularity_x = stride_regularity_x
        self.symmetry_x = stride_regularity_x - step_regularity_x

        self.step_regularity_y = step_regularity_y
        self.stride_regularity_y = stride_regularity_y
        self.symmetry_y = stride_regularity_y - step_regularity_y

        self.step_regularity_z = step_regularity_z
        self.stride_regularity_z = stride_regularity_z
        self.symmetry_z = stride_regularity_z - step_regularity_z
