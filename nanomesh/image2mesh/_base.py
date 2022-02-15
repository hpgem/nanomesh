from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from ..image import Plane, Volume
from ..mesh._base import BaseMesh

logger = logging.getLogger(__name__)


class BaseMesher(ABC):
    """Utility class to mesh image data and generate a mesh.

    Parameters
    ----------
    image : np.array
        N-dimensional numpy array containing image data.

    Attributes
    ----------
    image : np.ndarray
        Reference to image data
    image_orig : np.ndarray
        Keep reference to original image data
    contour : BaseMesh
        Stores the contour mesh.
    """

    def __init__(self, image: Union[np.ndarray, Plane, Volume]):
        if isinstance(image, (Plane, Volume)):
            image = image.image

        self.contour: BaseMesh | None = None
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
