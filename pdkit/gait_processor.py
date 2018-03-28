import sys
import traceback
import numpy as np

from .processor import Processor
from .utils import load_data, numerical_integration, autocorrelation, peakdet

from pywt import wavedec

class GaitProcessor(Processor):
    """Class used extract gait features from accelerometer data
    """
    def __init__(self, step_size=50.0, start_offset=100, end_offset=100, delta=0.5, loco_band=[0.5, 3], freeze_band=[3, 8]):
        super().__init__()

        self.freeze_time = None
        self.locomotion_freeze = None
        self.freeze_index = None

        self.step_size = step_size
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.delta = delta
        self.loco_band = loco_band
        self.freeze_band = freeze_band


    def freeze_of_gait(self, data_frame):
        """Following http://delivery.acm.org/10.1145/1660000/1658515/a11-bachlin.pdf
        """
        
        # the sampling frequency was recommended by the author of the pilot study
        data = self.resample_signal(data_frame) 
        data = data.y.values

        f_res = self.sampling_frequency / self.window
        f_nr_LBs = int(self.loco_band[0] / f_res)
        f_nr_LBe = int(self.loco_band[1] / f_res)
        f_nr_FBs = int(self.freeze_band[0] / f_res)
        f_nr_FBe = int(self.freeze_band[1] / f_res)

        jPos = self.window + 1
        i = 0
        
        time = []
        sumLocoFreeze = []
        freezeIndex = []
        
        while jPos < len(data):
            
            jStart = jPos - self.window
            time.append(jPos)

            y = data[int(jStart):int(jPos)]
            y = y - np.mean(y)

            Y = np.fft.fft(y, int(self.window))
            Pyy = abs(Y*Y) / self.window #conjugate(Y) * Y / NFFT

            areaLocoBand = numerical_integration( Pyy[f_nr_LBs-1 : f_nr_LBe], self.sampling_frequency )
            areaFreezeBand = numerical_integration( Pyy[f_nr_FBs-1 : f_nr_FBe], self.sampling_frequency )

            sumLocoFreeze.append(areaFreezeBand + areaLocoBand)

            freezeIndex.append(areaFreezeBand / areaLocoBand)

            jPos = jPos + self.step_size
            i = i + 1

        self.freeze_times = time
        self.freeze_indexes = freezeIndex
        self.locomotion_freezes = sumLocoFreeze


    def frequency_of_peaks(self, data_frame, delta=0.5):
        # this method calculatess the frequency from the peaks of the x-axis acceleration
        peaks_data = data_frame[self.start_offset:-self.end_offset].x.values
        self.peaks_data = peaks_data

        maxtab, mintab = peakdet(peaks_data, delta)

        x = np.mean(peaks_data[maxtab[1:,0].astype(int)] - peaks_data[maxtab[:-1,0].astype(int)])
        
        self.frequency_from_peaks = 1/x
        

    def speed_of_gait(self, data_frame, wavelet_type='db3', wavelet_level=6):
        # the technique followed in this method is described in detail in [2]
        # it involves wavelet transforming the signal and calculating
        # the gait speed from the energies of the approximation coefficients
        coeffs = wavedec(data_frame.mag_sum_acc, wavelet=wavelet_type, level=wavelet_level)

        energy = [sum(coeffs[wavelet_level - i]**2) / len(coeffs[wavelet_level - i]) for i in range(wavelet_level)]

        WEd1 = energy[0] / (5 * np.sqrt(2))
        WEd2 = energy[1] / (4 * np.sqrt(2))
        WEd3 = energy[2] / (3 * np.sqrt(2))
        WEd4 = energy[3] / (2 * np.sqrt(2))
        WEd5 = energy[4] / np.sqrt(2)
        WEd6 = energy[5] / np.sqrt(2)

        speed= 0.5 * np.sqrt(WEd1+(WEd2/2)+(WEd3/3)+(WEd4/4)+(WEd5/5))

        self.gait_speed = speed

    def walk_regularity_symmetry(self, data_frame):
        
        def _symmetry(v):
            maxtab, _ = peakdet(v, self.delta)
            return maxtab[1][1], maxtab[2][1]

        step_regularity_x, stride_regularity_x = _symmetry(autocorrelation(data_frame.x))
        step_regularity_y, stride_regularity_y = _symmetry(autocorrelation(data_frame.y))
        step_regularity_z, stride_regularity_z = _symmetry(autocorrelation(data_frame.z))

        symmetry_x = stride_regularity_x - step_regularity_x
        symmetry_y = stride_regularity_y - step_regularity_y
        symmetry_z = stride_regularity_z - step_regularity_z

        self.step_regularity = [step_regularity_x, step_regularity_y, step_regularity_z]
        self.stride_regularity = [stride_regularity_x, stride_regularity_y, stride_regularity_z]
        self.walk_symmetry = [symmetry_x, symmetry_y, symmetry_z]
