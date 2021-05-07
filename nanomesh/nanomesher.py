# -*- coding: utf-8 -*-"""Documentation about the nanomesh module."""
import logging
import os
from types import SimpleNamespace
from typing import List

import numpy as np
import SimpleITK as sitk

from .sitk_filter import (binary_threshold, confidence_connected,
                          gaussian_filtering, otsu_filtering, otsu_threshold)

logger = logging.getLogger(__name__)


class NanoMesher(object):
    """Main Class to mesh the data."""
    def __init__(self):
        """Initialize the class.

        Args:
            data_file_name (str): file name of the data to load
        """

        self.volume = SimpleNamespace(name=None,
                                      info=None,
                                      size=None,
                                      format=None,
                                      data=None,
                                      img=None)

    def load_numpy(self, filename: str):
        """Load from a numpy file.

        Args:
            filename (str): filename
        """
        self.volume.name = filename
        self.volume.info = None
        self.volume.data_format = 'numpy'
        self.volume.data = np.loadtxt(filename)
        self.normalize_data()
        self.volume.size = self.volume.data.shape
        self.create_image()

    def load_bin(self,
                 data_file_name: str,
                 info_file_name: str = None,
                 size: List = [],
                 input_dtype=np.float32,
                 output_dtype=np.float32,
                 create_image=True,
                 rescale=False):
        """Load from a binary file.

        Args:
            data_file_name (str): file of the data
            info_file_name (str, optional): file describing the data.
                Defaults to None.
            size (List, optional): size. Defaults to [].
        """
        if info_file_name is None and size == []:
            raise ValueError(
                'You must provide either an info file or the size of the data')

        self.volume.name = data_file_name

        if info_file_name is not None:
            if os.path.isfile(info_file_name):
                self.read_info(info_file_name)
                self.volume.size = [
                    self.volume.info['NUM_Z'], self.volume.info['NUM_Y'],
                    self.volume.info['NUM_X']
                ]
            else:
                raise FileNotFoundError(info_file_name)
        else:
            self.volume.size = size

        logging.info('Processing file : ', self.volume.name)
        with open(self.volume.name, 'rb') as fid:
            self.volume.data = np.fromfile(fid, dtype=input_dtype)

        self.volume.data = self.volume.data.reshape(self.volume.size)

        if self.volume.data.dtype != output_dtype:
            self.volume.data = self.volume.data.astype(output_dtype)

        if create_image:
            self.create_image()

        if rescale:
            self.normalize_data()

    def read_info(self, info_file: str):
        """Load the info about the data."""
        def str2data(str_val: str):
            str_val = str_val.strip()
            if '.' in str_val:
                return float(str_val)
            else:
                try:
                    int_val = int(str_val)
                except BaseException:
                    return str_val
                else:
                    return int_val

        self.volume.info = {'info_file_name': info_file}
        with open(info_file, 'r') as fid:
            for line in fid:
                if '=' in line:
                    (key, val) = line.split('=')
                    self.volume.info[key.strip()] = str2data(val)

    def normalize_data(self):
        """Normalize the data from  0 to 1."""
        self.volume.data += self.volume.data.min()
        self.volume.data /= self.volume.data.max()

    def create_image(self, data=None, rescale=True):
        """Create the sitk image.

        Args:
            rescale (bool, optional): rescale the intensity. Defaults to True.
        """
        if data is None:
            self.volume.img = sitk.GetImageFromArray(self.volume.data)
        else:
            self.volume.img = sitk.GetImageFromArray(data)

        if rescale:
            self.volume.img = sitk.Cast(sitk.RescaleIntensity(self.volume.img),
                                        sitk.sitkUInt8)

    def apply_otsu_filtering(self, rescale=True):
        """Apply otsu."""
        self.volume.img = otsu_filtering(self.volume.img, rescale=rescale)

    def apply_gaussian_filtering(self, sigma=2., rescale=True):
        """Apply gaussian filter."""
        self.volume.img = gaussian_filtering(self.volume.img,
                                             sigma=sigma,
                                             rescale=rescale)

    def apply_binary_threshold(self,
                               lowerThreshold=100,
                               upperThreshold=150,
                               insideValue=1,
                               outsideValue=0):
        """apply binary threshold."""
        self.volume.img = binary_threshold(self.volume.img,
                                           lowerThreshold=lowerThreshold,
                                           upperThreshold=upperThreshold,
                                           insideValue=insideValue,
                                           outsideValue=outsideValue)

    def apply_otsu_threshold(self, insideValue=1, outsideValue=0):
        """apply otsu threshold."""
        self.volume.img = otsu_threshold(self.volume.img,
                                         insideValue=insideValue,
                                         outsideValue=outsideValue)

    def apply_confidence_connected(self,
                                   seed,
                                   numberOfIterations=1,
                                   multiplier=2.5,
                                   initialNeighborhoodRadius=1,
                                   replaceValue=1):
        """apply confidence connected threshold."""
        self.volume.img = confidence_connected(
            self.volume.img,
            seed,
            numberOfIterations=numberOfIterations,
            multiplier=multiplier,
            initialNeighborhoodRadius=initialNeighborhoodRadius,
            replaceValue=replaceValue)
