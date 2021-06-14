# -*- coding: utf-8 -*-
"""Documentation about nanomesh."""
import logging
import sys

from .__version__ import __version__
from .plane import Plane
from .volume import Volume

logging.basicConfig(format='%(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

__author__ = 'Nicolas Renaud'
__email__ = 'n.renaud@esciencecenter.nl'

__all__ = [
    '__author__',
    '__email__',
    '__version__',
    'Plane',
    'Volume',
]
