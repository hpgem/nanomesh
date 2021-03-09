from nanomesh.generator import Generator
import SimpleITK as sitk
from nanomesh.utils import show_slice, show_volume, generate_mesh_from_binary_image
import numpy as np
import math 
import pygalmesh    

class Cross(pygalmesh.DomainBase):
    def __init__(self):
        super().__init__()
        
        self.npts = 3   
        self.size = 1
        self.x = np.linspace(0,self.size,self.npts)
        self.data = np.ones((self.npts,self.npts,self.npts))
        
        
        # idx = (self.x>0.25*self.size) & (self.x<0.75*self.size)
        # all_idx = list(range(self.npts))
        # self.data[np.ix_(idx,idx,all_idx)] = 0.
        # self.data[np.ix_(idx,all_idx,idx)] = 0.
        # self.data[np.ix_(all_idx,idx,idx)] = 0.
                    

    def eval(self, x):
        ix = np.argmin(np.abs(x[0]-self.x))
        iy = np.argmin(np.abs(x[1]-self.x))
        iz = np.argmin(np.abs(x[2]-self.x))
        if self.data[ix,iy,iz] == 1:
            return -1.
        else:
            return 0.

# gen = Generator(680, 680*math.sqrt(2), 0.24*680)

# # Possible rotation/transformation of the coordinate system
# theta = math.pi * 1/180
# c = math.cos(theta)
# s = math.sin(theta)
# trans = np.array([
#     [ c, 0, s],
#     [ 0, 1, 0],
#     [-s, 0, c]
# ])

# vol = gen.generate_vect([100]*3, [10]*3, transform=trans, bin_val=[0.,1.])
# im = sitk.GetImageFromArray(vol.astype('uint8'))


cross = Cross()
im = sitk.GetImageFromArray(cross.data.astype('uint8'))

mesh = generate_mesh_from_binary_image(im, h=[1.]*3, perturb=False, 
                                       lloyd=False,
                                       odt=False,
                                       max_radius_surface_delaunay_ball=0.0,
                                       max_cell_circumradius=0.,
                                       max_facet_distance=1.,
                                       exude=True,
                                       min_facet_angle=0.,
                                       max_edge_size_at_feature_edges=1.5,
                                       max_circumradius_edge_ratio=0.,
                                    #    extra_feature_edges=[[[0.0,0.0,0.],[100.0,0.0,0.]],[[100.0,100.0,0.],[100.,0.0,0.]]],
                                    #    extra_feature_edges=[[[0.0,0.0,0.],[100.0,0.0,0.],[100.0,100.0,0.],[0.,100.0,0.]]],
                                       bbox_feature=True)
