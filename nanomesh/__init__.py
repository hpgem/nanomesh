# -*- coding: utf-8 -*-
"""Documentation about nanomesh."""
import logging

from .__version__ import __version__
from .plane import Plane
from .volume import Volume

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = 'Nicolas Renaud'
__email__ = 'n.renaud@esciencecenter.nl'

__all__ = [
    '__author__',
    '__email__',
    '__version__',
    'Plane',
    'Volume',
]
