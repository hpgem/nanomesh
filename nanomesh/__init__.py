# -*- coding: utf-8 -*-
"""Documentation about nanomesh."""
import logging
import sys

from .plane import Plane
from .volume import Volume

logging.basicConfig(format='%(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

__author__ = 'Nicolas Renaud'
__email__ = 'n.renaud@esciencecenter.nl'
__version__ = '0.1.0'

__all__ = [
    '__author__',
    '__email__',
    '__version__',
    'Plane',
    'Volume',
]
