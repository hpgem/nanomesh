---
title: 'Nanomesh: A Python workflow tool for generating meshes from image data '
tags:
  - Python
  - finite-element-methods
  - microscopy
  - meshing
  - image-processing
authors:
  - name: Stef Smeets^[Corresponding author]
    orcid: 0000-0002-5413-9038
    affiliation: 1
  - name: Nicolas Renaud
    orcid: 0000-0001-9589-2694
    affiliation: 1
  - name: Lars J. Corbijn van Willenswaard
    affiliation: 2
affiliations:
 - name: Netherlands eScience Center, The Netherlands
   index: 1
 - name: University of Twente, The Netherlands
   index: 2
date: 9 May 2022
bibliography: paper.bib
---

# Summary

Nanomesh is a Python library that allows users to quicly and easily create 2D and 3D meshes directly from images of the object they wish to mesh. The automated workflow can apply filtering to smooth the original image, segment the picture to extract different regions and create conforming meshes of the objects. Analysis tools allow to evaluate the quality of the resuting mesh and detect problematic regions. The resulting meshes can be exproted in a variety of popular formats so that they can easily be used for example in finite element simulations. nanomesh can be used as python library for example in Jupyter notebooks, or though dedicated online dashboards.

# Statement of need

Finite element simulations very often require the creation of a non-regular mesh that represent the topology and physical properties of the object under examination.  Many meshing libraries exists and allows to create such meshes [ref]. Most of these solutions are however proprietary with sometimes a substantial fee due to the level of certification they require for example for medical or engineering applications [ref].  Some open source libraries do exist but often create meshes from a CAD design or well defined primitive [ref]. While these meshing libraries are invaluable for the study of idealized systems they do not allow the mesh to account for potential defects in the underlying topology of the object. 

For example the calcuations of the optical properties of nanocrystals is usually performed with an ideal nano structure as substrate for the propagation of the Maxwell equations [ref]. Such simulations provide very valuable insight but ignore the effect that manufacturing imprecision of the nanometer size pores can have on the overall properties of the crystal. To resolve such structure-property relationship, a meshes conforming to experimental images of real nanocrystlals are needed. The subsequent simulation of wave propagation through  these meshes using any FEM solver will enable to better understand the impact that imperfections may have on the overall properties. Similar use cases in different fields of material science and beyond are expected. The direct FEM simulations on real device topology might bring very valuable insights. Through its user friendliness, code qualitiy, nanomesh will enable scientist running advanced simulations on mesh that represent accurately the  devices that are created in experimental labs.

# Illustration 

To illustrate the use of nanomesh we present here the use of nanomesh to create 2D and 3D meshes of nanopores etched in a silicon matrix. These nanopores are very often used in the creation of optical crystals and the study of their properties is therefore crucial. [Or maybe the cell example] 

<!-- # Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text. -->

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from ...

# References
