#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2019 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author: Cosmin Stamate
import shutil
import os
import logging
import sys
import urllib.request

import numpy as np

import soundfile as sf
import parselmouth
from parselmouth.praat import call, run_file


class VoiceProcessor:
    """
        Voice processing for OPDC and Hopkins PD RAW adio data.
        Computations are carried out by Praat via the pytohn parcelmount module.
        Additional Praat computation plan from the python myprosody module.

    """
    def __init__(self, filename, format_file='opdc'):

                try:
                    vdata, samplerate = sf.read(filename, channels=1, samplerate=44100, subtype='PCM_16')
                    # offset=int(len(data)/20)
                    # d_crop=data[offset:12*offset]
                    prefix, extension = os.path.splitext(filename)
                    self.file_name = prefix + ".wav"
                    sf.write(self.file_name, vdata, format='WAV', samplerate=48000, subtype='FLOAT')

                    logging.debug("VoiceProcessor init")

                except IOError as e:
                    ierr = "({}): {}".format(e.errno, e.strerror)
                    logging.error("VoiceProcessor I/O error %s", ierr)

                except ValueError as verr:
                    logging.error("VoiceProcessor ValueError ->%s", verr.message)

                except:
                    logging.error("Unexpected error in VoiceProcessor init: %s", sys.exc_info()[0])


    def extract_features(self, pre=''):
        """
            This method extracts all the features available in the Voice Processor class.

            :param pre: feature name prefix
            :type string: string
            :return: f0_mean, f0_SD, f0_MD, f0_min, f0_max, f0_quan25, f0_quan75, n_pulses, n_periods, shimmer_local, median
            :rtype: list

        """
        try:

            # urllib.request.urlretrieve("https://raw.githubusercontent.com/Shahabks/myprosody/master/myprosody/dataset/essen/myspsolution.praat", "myspsolution.praat")
            objects=run_file(os.getcwd() + "/pdkit/myspsolution.praat", -20, 2, 0.3, "yes", self.file_name, './', 80, 400, 0.01, capture_output=True)
            if 'No result' not in str(objects):
                z1=str( objects[1])
                z2=z1.strip().split()

                f0_mean=float(z2[7])
                f0_SD=float(z2[8])
                f0_MD=float(z2[9])
                f0_min=float(z2[10])
                f0_max=float(z2[11])
                f0_quan25=float(z2[11])
                f0_quan75=float(z2[11])

                sound = parselmouth.Sound(self.file_name)
                pitch = sound.to_pitch()
                pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
                n_pulses = parselmouth.praat.call(pulses, "Get number of points")
                n_periods = parselmouth.praat.call(pulses, "Get number of periods", 0.0, 0.0, 0.0001, 0.02, 1.3)
                shimmer_local = parselmouth.praat.call([sound, pulses], "Get shimmer (local)...", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
                max_voiced_period = 0.02  # This is the "longest period" parameter in some of the other queries
                periods = [parselmouth.praat.call(pulses, "Get time from index", i+1) -
                           parselmouth.praat.call(pulses, "Get time from index", i)
                           for i in range(1, n_pulses)]
                median=parselmouth.praat.call(pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz")

                # print(f0_mean, f0_SD, f0_MD, f0_min, f0_max, f0_quan25, f0_quan75, n_pulses, n_periods, shimmer_local, median)

                return {pre+'f0_mean': f0_mean,
                        pre+'f0_SD': f0_SD,
                        pre+'f0_MD': f0_MD,
                        pre+'f0_min': f0_min,
                        pre+'f0_max': f0_max,
                        pre+'f0_quan25': f0_quan25,
                        pre+'f0_quan75': f0_quan75,
                        pre+'n_pulses': n_pulses,
                        pre+'n_periods': n_periods,
                        pre+'shimmer_local': shimmer_local,
                        pre+'median': median}
            else:
                return {pre+'f0_mean': float('NaN'),
                        pre+'f0_SD': float('NaN'),
                        pre+'f0_MD': float('NaN'),
                        pre+'f0_min': float('NaN'),
                        pre+'f0_max': float('NaN'),
                        pre+'f0_quan25': float('NaN'),
                        pre+'f0_quan75': float('NaN'),
                        pre+'n_pulses': float('NaN'),
                        pre+'n_periods': float('NaN'),
                        pre+'shimmer_local': float('NaN'),
                        pre+'median': float('NaN')}

        except:
            logging.error("Error on VoiceProcessor extract features: %s", sys.exc_info()[0])
