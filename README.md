Metric deformation by Ricci flow
==========

- ricci_flow.py:

[Original](https://github.com/hchapman/ricci-flow) written by Harrison Chapman.
Bug fix/modified by S. Kaji (Dec. 2020)

- metric_embed.py:
written by S. Kaji

# Requirements

    pip install plyfile

# How to use

    python ricci_flow.py -i torus.obj -m thurston -uh

reads mesh from torus.obj and deform its metric (edge length) by Ricci flow on Thurston's Circle Packing metric.

    python ricci_flow.py -i torus.obj -m unified -uh

uses the inversive Circle Packing metric.

The output result/torus_edge.csv contains edge length per line:

    i,j,length of edge connecting vertex i and j

The PLY file result/torus.ply is same as the (non-deformed) original mesh. 
These two files are then passed to 

    python metric_embed.py result/torus.ply -lv 0 -fb

to obtain the final deformed mesh result/torus_final.ply.

Curvature distributions in violin plot are provided in png files.


