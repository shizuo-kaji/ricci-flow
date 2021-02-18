Metric deformation by Ricci flow
==========
written by S. Kaji
based on [Original](https://github.com/hchapman/ricci-flow) written by Harrison Chapman.

This program takes 
- a surface mesh 
- a specification of Gaussian curvature (angle defect) for each vertex
- boundary conditions
- a conformal class specified by circle packing or by the inigial mesh
and produces a surface mesh meeting the conditions.

In contrast to usual implementations of Ricci flow based geometry processing,
this code directly optimises the Ricci energy, and hence, offers a flexibility of
incorpolating various constraints.
Also, as our scheme is based on a simple optimisation problem of a scalar function,
any advanced optimisers can be easily utilised.

# Requirements

- Python 3: [Anaconda](https://anaconda.org) is recommended
- plyfile installed by

    pip install plyfile

# How to use

## with boundary and specified target curvatures for each vertex

    python ricci_flow.py dome.ply -K dome_targetK_hat.csv --gtol 1e-6 -op trf

reads mesh from dome.ply and target vertex curvatures from dome_targetK_hat.csv and performs Ricci flow.
(for vertices with target curvature > 2pi or with no specified target curvature values, the target curvatures are inferred by the Gauss-Bonnet theorem).
The trf optimiser with gtol 1e-6 is used (see scipy.optimize).

Each row of dome_targetK_hat.csv consists:

    i, Guassian curvature of vertex i


The output result/dome_edge.csv contains edge length per line:

    i,  j,  length of edge connecting vertex i and j

The PLY file result/dome.ply is same as the (non-deformed) original mesh. 
These two files are then passed to 

    python metric_embed.py result/dome.ply -lv 0 --gtol 1e-6

to obtain the final deformed mesh result/dome_final.ply.
Curvature distributions in violin plot are provided in png files.
Furthermore, 

    python evaluation.py result/dome_final.ply

produces various error statistics.

## with boundary fixed and uniform curvature for interior vertices

    python ricci_flow.py dome.ply -lb -Ki 0.1

reads mesh from dome.ply and performs Ricci flow with target curvature 0.1 for interior vertices, while leaving boundary vertices intact.

The final result (mesh) is obtained by

    python metric_embed.py result/dome.ply --gtol 1e-6

## with boundary fixed, uniform curvature for interior vertices, and as regular as possible

    python ricci_flow.py dome.ply -lb -Ki 0.1 -m combinatorial

reads mesh from dome.ply and performs Ricci flow with target curvature 0.1 for interior vertices, while leaving boundary vertices intact.
Instead of preserving the shape of faces, the metric is sought to make the faces as close as possible to equilateral triangles.

The final result (mesh) is obtained by

    python metric_embed.py result/dome.ply --gtol 1e-6


## specified uniform curvature for interior vertices and uniform curvature for boundary

    python ricci_flow.py dome.ply -Ki 0 

reads mesh from planar_grid.ply and performs Ricci flow with target curvature 0 for interior vertices and inferred uniform target curvature for boundary.

The final result is obtained by

    python metric_embed.py result/dome.ply -lv 0




## without boundary

    python ricci_flow.py torus.obj -m inversive -op newton

reads mesh from torus.obj and deform its metric (edge length) by Ricci flow on inversive Circle Packing metric.

    python ricci_flow.py torus.obj -m thurston -op newton

uses the Thurston's Circle Packing metric.

    python ricci_flow.py torus.obj -m combinatorial -op newton

ignores the initial geometry and construct a circle packing purely combinatorially from the mesh.

The PLY file result/torus.ply is same as the (non-deformed) original mesh. 
The final result is obtained by

    python metric_embed.py result/torus.ply -lv 0 -fb --gtol 1e-6

to obtain the final deformed mesh result/torus_final.ply.


## Schemes for circle packing metric

Currently, thurston, thurston2, inversive, combinatorial are supported.

For example,

    python ricci_flow.py dome.ply -lb -Ki 0 -m 1.0 -e

yields a final mesh dome_final.ply in the conformal class with a constant edge weight 1.0.

## Limitation

Ricci flow converges to give a metric with target cuvatures if the target curvatures are _admissible_.


# Files

The following intermediate files are produced by ricci_flow.py:
- *_boundary.csv containing fixed boundary coordinates; each row consists of

    id,  x,   y,   z

- *_edge.csv containing edge lengths computed by the Ricci flow; each row consists of

    id,  id,  length

- *_targetCurvature.txt containing the target Gaussian curvature values for vertices
- *_innerVertexID.txt containing the list of IDs of interior vertices
- *_curvature_ricci.png showing the distribution of the curvature of free vertices computed from the metric found by the ricci flow

The following intermediate files are produced by metric_embed.py:
- *_edge_scaled.csv containing the uniformly scaled edge length found to meet the boundary constraint: it is obtained by multiplying a constant to values in *_edge.csv.
- *_curvature_final.png showing the distribution of the curvature of free vertices of the final mesh

