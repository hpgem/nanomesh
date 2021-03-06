import SimpleITK as sitk
import matplotlib.pyplot as plt

from ipywidgets import interact, interactive
from ipywidgets import widgets

import itkwidgets as itkw
import pygalmesh
import meshio

def show_slice(img, dim='x', title=None, scale = 1., margin=0.05, dpi=80 ):

    if isinstance(img, tuple):
        img = sitk.LabelOverlay(*img)

    nda = sitk.GetArrayFromImage(img)

    spacing = img.GetSpacing()
    slicer = False

    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]

        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3,4):
            slicer = True

    elif nda.ndim == 4:
        c = nda.shape[-1]

        if not c in (3,4):
            raise RuntimeError("Unable to show 3D-vector Image")

        # take a z-slice
        slicer = True

    if (slicer):
        ysize = nda.shape[1]
        xsize = nda.shape[2]
    else:
        ysize = nda.shape[0]
        xsize = nda.shape[1]


    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = scale * (1 + margin) * ysize / dpi, scale * (1 + margin) * xsize / dpi
    def callback(z=None):

        extent = (0, xsize*spacing[1], ysize*spacing[0], 0)

        fig = plt.figure(figsize=figsize, dpi=dpi)

        # Make the axis the right size...
        ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

        plt.set_cmap("gray")

        if z is None:
            ax.imshow(nda,extent=extent,interpolation=None)
        else:
            if dim == 'x':
                ax.imshow(nda[z,...],extent=extent,interpolation=None)
            elif dim == 'y':
                ax.imshow(nda[:,z,:],extent=extent,interpolation=None)
            if dim == 'z':
                ax.imshow(nda[:,:,z],extent=extent,interpolation=None)

        if title:
            plt.title(title)

        plt.show()

    if slicer:
        interact(callback, z=(0,nda.shape[0]-1))
    else:
        callback()


def show_volume(data):
    return itkw.view(data)

def generate_mesh_from_binary_image(img, h=[1.]*3, **kwargs):
    img_array = sitk.GetArrayFromImage(img)
    mesh = pygalmesh.generate_from_array(img_array, h, **kwargs)
    return mesh

    
    


