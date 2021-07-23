import logging
from collections import defaultdict
from typing import Dict, Union

import numpy as np
from sklearn import cluster, mixture

from nanomesh import Plane, Volume

logger = logging.getLogger(__name__)


def add_points_kmeans(image: np.ndarray,
                      iters: int = 10,
                      n_points: int = 100,
                      label: int = 1,
                      **kwargs):
    """Add evenly distributed points to the image using K-Means clustering.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image
    iters : int, optional
        Number of iterations for the algorithm.
    n_points : int, optional
        Total number of points to add
    label : int, optional
        Domain to select coordinates from
    **kwargs
        Extra keyword arguments to pass to `sklearn.cluster.KMeans`

    Returns
    -------
    (n,2) np.ndarray
        Array with the generated points.
    """
    kwargs.setdefault('max_iter', iters)

    coordinates = np.argwhere(image == label)

    kmeans = cluster.KMeans(n_clusters=n_points,
                            n_init=1,
                            init='random',
                            algorithm='full',
                            **kwargs)
    ret = kmeans.fit(coordinates)

    return ret.cluster_centers_


def add_points_gaussian_mixture(image: np.ndarray,
                                iters: int = 10,
                                n_points: int = 100,
                                label: int = 1,
                                **kwargs):
    """Add evenly distributed points to the image using a Gaussian Mixture
    model.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image
    iters : int, optional
        Number of iterations for the algorithm.
    n_points : int, optional
        Total number of points to add
    label : int, optional
        Domain to select coordinates from
    **kwargs
        Extra keyword arguments to pass to `sklearn.mixture.GaussianMixture`

    Returns
    -------
    (n,2) np.ndarray
        Array with the generated points.
    """
    kwargs.setdefault('max_iter', iters)

    coordinates = np.argwhere(image == label)

    gmm = mixture.GaussianMixture(n_components=n_points, **kwargs)
    ret = gmm.fit(coordinates)

    return ret.means_


class BaseMesher:
    def __init__(self, image: Union[np.ndarray, Plane, Volume]):
        if isinstance(image, (Plane, Volume)):
            image = image.image

        self.image_orig = image
        self.image = image
        self.points: Dict[int, list] = defaultdict(list)

    def add_points(
        self,
        point_density: float,
        *,
        label: int = 1,
        method: str = 'kmeans',
        **kwargs,
    ):
        """Generate evenly distributed points using K-Means in the domain body
        for generating tetrahedra. Alternative implementation using a Gaussian
        Mixture model available via `method='gmm'`.

        Parameters
        ----------
        point_density : float, optional
            Density of points (points per pixels) to distribute over the
            domain for triangle generation.
        label : int, optional
            Label of the domain to add points to.
        method : str
            Clustering algorithm to use,
                `kmeans` : `sklearn.cluster.KMeans`
                `gmm` : `sklearn.mixture.GaussianMixture`
        **kwargs :
            Keywords arguments for the clustering algorithm.
        """
        n_points = int(np.sum(self.image == label) * point_density)

        if method == 'kmeans':
            add_points_func = add_points_kmeans
        elif method == 'gmm':
            add_points_func = add_points_gaussian_mixture
        else:
            raise ValueError(f'Unknown method: {method!r}')
        grid_points = add_points_func(self.image,
                                      iters=10,
                                      n_points=n_points,
                                      label=label,
                                      **kwargs)
        self.points[label].append(grid_points)
        logger.info(f'added {len(grid_points)} points ({label=}), '
                    f'{point_density=}, {method=}')

    @property
    def flattened_points(self) -> list:
        """Return flattened list of pointss."""
        flat_list = [
            points for points_subset in self.points.values()
            for points in points_subset
        ]
        return flat_list
