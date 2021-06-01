# -*- coding: utf-8 -*-
"""Documentation about nanomesh."""
import logging

from .__version__ import __version__
from .utils import show_slice, show_volume

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = 'Nicolas Renaud'
__email__ = 'n.renaud@esciencecenter.nl'

__all__ = [
    '__version__',
    '__author__',
    '__email__',
    'NanoMesher',
    'show_slice',
    'show_volume',
]
