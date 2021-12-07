import logging
from typing import Union

import numpy as np

from .plane import Plane
from .volume import Volume

logger = logging.getLogger(__name__)


class BaseMesher:
    def __init__(self, image: Union[np.ndarray, Plane, Volume]):
        if isinstance(image, (Plane, Volume)):
            image = image.image

        self.image_orig = image
        self.image = image
