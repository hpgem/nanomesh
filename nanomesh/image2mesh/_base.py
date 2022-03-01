from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from .._doc import doc
from ..image import Plane, Volume
from ..mesh._base import GenericMesh

logger = logging.getLogger(__name__)


@doc(prefix='mesh from image data')
class AbstractMesher:
    """Utility class to generate a {prefix}.

    Parameters
    ----------
    image : np.array
        N-dimensional numpy array containing image data.

    Attributes
    ----------
    image : numpy.ndarray
        Reference to image data
    image_orig : numpy.ndarray
        Keep reference to original image data
    contour : GenericMesh
        Stores the contour mesh.
    """
    _registry = {}

    def __init_subclass__(cls, ndim, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[ndim] = cls

    def __new__(cls, image):
        ndim = image.ndim
        subclass = cls._registry.get(ndim, cls)
        obj = object.__new__(subclass)
        obj.__init__(image)
        return obj

    def __init__(self, image: Union[np.ndarray, Plane, Volume]):
        if isinstance(image, (Plane, Volume)):
            image = image.image

        self.contour: GenericMesh | None = None
        self.image_orig = image
        self.image = image

    def __repr__(self):
        """Canonical string representation."""
        s = (
            f'{self.__class__.__name__}(',
            f'    image = {self.image!r},',
            f'    contour = {self.contour.__repr__(indent=4)}'
            ')',
        )
        return '\n'.join(s)

    @abstractmethod
    def plot_contour(self):
        ...

    @abstractmethod
    def generate_contour(self):
        ...
