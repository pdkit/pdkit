import sys
import traceback
import numpy as np

from .processor import Processor

from .utils import estimate_autocorrelation, numerical_integration

from scipy import interpolate, signal, fft
from pywt import wavedec

class GaitProcessor(Processor):
    '''
       This is the main Gait Processor class. Once the data is loaded it will be
       accessible at data_frame, where it looks like:
       data_frame.x, data_frame.y, data_frame.z: x, y, z components of the acceleration
       data_frame.index is the datetime-like index
       
       This values are recommended by the author of the pilot study [1]
       
       sampling_frequency = 100.0Hz
       cutoff_frequency = 2.0Hz
       filter_order = 2
       window = 256
       lower_frequency = 2.0Hz
       upper_frequency = 10.0Hz

       [1] Developing a tool for remote digital assessment of Parkinson s disease
            Kassavetis	P,	Saifee	TA,	Roussos	G,	Drougas	L,	Kojovic	M,	Rothwell	JC,	Edwards	MJ,	Bhatia	KP
            
       [2] The use of the fast Fourier transform for the estimation of power spectra: A method based 
            on time averaging over short, modified periodograms (IEEE Trans. Audio Electroacoust. 
            vol. 15, pp. 70-73, 1967)
            P. Welch
    '''

    def __init__(self):
        self.freeze_time = None
        self.locomotion_freeze = None
        self.freeze_index = None
    

    def detect_fog(self, sample_rate=100.0, step_size=50.0):
        '''
            F
        
        
        Following http://delivery.acm.org/10.1145/1660000/1658515/a11-bachlin.pdf
        '''
        
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


    def frequency_from_peaks(self, start_offset=100, end_offset=100, delta=0.5):
        # this method calculatess the frequency from the peaks of the x-axis acceleration
        self.peaks_data_frame = self.data_frame[start_offset:-end_offset]

        maxtab, mintab = np.peakdet(self.peaks_data_frame.x, delta)

        x = np.mean(self.peaks_data_frame.dt[maxtab[1:,0].astype(int)] - self.peaks_data_frame.dt[maxtab[0:-1,0].astype(int)])
        
        self.frequency_from_peaks = 1/x


    def gait_speed(self, wavelet_type='db3', wavelet_level=6):
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

    def regularity_symmetry(self):
        
        def _symmetry(v):
            # DELTA = 0.5 as per Matlab code for gait analysis
            maxtab, _ = np.peakdet(v, DELTA=0.5)
            return maxtab[1][1], maxtab[2][1]

        step_regularity_x, stride_regularity_x = _symmetry(self.estimated_autocorrelation(self.data_frame.x))
        step_regularity_y, stride_regularity_y = _symmetry(self.estimated_autocorrelation(self.data_frame.x))
        step_regularity_z, stride_regularity_z = _symmetry(self.estimated_autocorrelation(self.data_frame.x))

        self.step_regularity_x = step_regularity_x
        self.stride_regularity_x = stride_regularity_x
        self.symmetry_x = stride_regularity_x - step_regularity_x

        self.step_regularity_y = step_regularity_y
        self.stride_regularity_y = stride_regularity_y
        self.symmetry_y = stride_regularity_y - step_regularity_y

        self.step_regularity_z = step_regularity_z
        self.stride_regularity_z = stride_regularity_z
        self.symmetry_z = stride_regularity_z - step_regularity_z
