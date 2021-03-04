from nanomesh.generator import Generator
import numpy as np
import math

gen = Generator(680, 680*math.sqrt(2), 0.24*680)

# Possible rotation/transformation of the coordinate system
theta = math.pi * 1/180
c = math.cos(theta)
s = math.sin(theta)
trans = np.array([
    [ c, 0, s],
    [ 0, 1, 0],
    [-s, 0, c]
])

vol = gen.generate_vect([10]*3, [1]*3, transform=trans, bin_val=[0.,1.])