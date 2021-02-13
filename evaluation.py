#!/usr/bin/env python
import argparse,os,time
import seaborn as sns
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import numpy as np
from ricci_flow import DiscreteRiemannianMetric, createMesh

#########################
parser = argparse.ArgumentParser(description='embedding of metric graphs')
parser.add_argument('input', help='final mesh')
parser.add_argument('--boundary_vertex', '-bv', default=None, help='Path to a csv specifying boundary position')
parser.add_argument('--constrained_vert', '-cv', default=None, type=str, help='file containing indices of vertices with curvature target')
parser.add_argument('--initial_mesh', '-im', default=None, help='initial mesh')
parser.add_argument('--edge_length', '-el', default=None, help='Path to a csv specifying edge length')
parser.add_argument('--index_shift', type=int, default=1, help="vertex indices start at")
parser.add_argument('--outdir', '-o', default='result',help='Directory to output the result')
parser.add_argument('--target_curvature', '-K', default=None, type=str, help='file specifying target gaussian curvature')
parser.add_argument('--target_curvature_scalar', '-Ks', default=0.01181102, type=float, help='target gaussian curvature value')
args = parser.parse_args()

os.makedirs(args.outdir,exist_ok=True)

# Read mesh data
fn, ext = os.path.splitext(args.input)
pn, fn = os.path.split(fn)
fn = os.path.join(pn,os.path.basename(fn).rsplit('_', 1)[0])
if args.edge_length is None:
    args.edge_length = fn+"_edge_scaled.csv"
if args.boundary_vertex is None:
    cfn = fn+"_boundary.csv"
    if os.path.isfile(cfn):
        args.boundary_vertex= cfn
if args.constrained_vert is None:
    cfn = fn+"_innerVertexID.txt"
    if os.path.isfile(cfn):
        args.constrained_vert = cfn
if args.initial_mesh is None:
    cfn = fn+".ply"
    if os.path.isfile(cfn):
        args.initial_mesh = cfn
if args.target_curvature is None:
    cfn = fn+"_targetCurvature.txt"
    if os.path.isfile(cfn):
        args.target_curvature = cfn

print(args)
plydata = PlyData.read(args.input)
vert = np.vstack([plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z']]).T
#    vert2 = np.loadtxt(fn+"_init.txt")
#    print(np.abs(vert-vert2).sum())
face = plydata['face']['vertex_indices']

edgedat = np.loadtxt(args.edge_length,delimiter=",")
edgedat = np.array([ [i,j,l] for i,j,l in zip(edgedat[:,0],edgedat[:,1],edgedat[:,2]) if i<j ])
edgedict = {(i,j): l for i,j,l in zip(edgedat[:,0],edgedat[:,1],edgedat[:,2])}
inedge = edgedat[:,:2].astype(np.uint32)
edgelen = edgedat[:,2]
#dmat = np.sum((np.expand_dims(vert,axis=0) - np.expand_dims(vert,axis=1))**2,axis=2)
#dmat[inedge[:,0],inedge[:,1]] = edgelen
#dmat[inedge[:,1],inedge[:,0]] = edgelen
#    np.savetxt("dmat.txt",dmat)

if args.boundary_vertex:
    print("reading boundary data from ", args.boundary_vertex)
    bddat = np.loadtxt(args.boundary_vertex,delimiter=",")
    fixed_vert = bddat[:,0].astype(np.uint32)
    fixed_coords = bddat[:,1:]
else:
    fixed_vert =np.array([0])
    fixed_coords = vert[fixed_vert]

# set target curvature
if args.target_curvature:
    print("reading target curvature from ", args.target_curvature)
    targetK = np.loadtxt(args.target_curvature)
else:
    targetK = np.full(len(vert),args.target_curvature_scalar)

#
args.free_vert = list(set(range(len(vert))) - set(fixed_vert))
if args.constrained_vert:
    print("reading inner vertex indices from ", args.constrained_vert)
    constrained_vert = np.loadtxt(args.constrained_vert).astype(np.uint16)
else:
    constrained_vert = list(set(args.free_vert) - set(np.where( targetK == -99 )[0]))

print("\nvertices {}, faces {}, fixed vertices {}, K-constrained {}".format(len(vert),len(face),len(fixed_vert),len(constrained_vert)))

mesh_final = createMesh(vert,[frozenset(x) for x in face])
g_final = DiscreteRiemannianMetric(mesh_final, mesh_final.lengths)
g_dmat = DiscreteRiemannianMetric(mesh_final, edgedict)

## conformality
if args.initial_mesh is not None:
    plydata_init = PlyData.read(args.initial_mesh)
    vert_init = np.vstack([plydata_init['vertex']['x'],plydata_init['vertex']['y'],plydata_init['vertex']['z']]).T
    face_init = plydata_init['face']['vertex_indices']
    mesh_init = createMesh(vert_init,[frozenset(x) for x in face_init])
    g_init = DiscreteRiemannianMetric(mesh_init, mesh_init.lengths)
    angles_init = g_init.angle_array()
    angles_final = g_final.angle_array()
    angles_dmat = g_dmat.angle_array()
    print("angle squared difference: init vs final {}, init vs dmat {}".format(((angles_init-angles_final)**2).sum(),  ((angles_init-angles_dmat)**2).sum()))


# %%
np.savetxt(os.path.join(args.outdir,"curvature.txt"),np.vstack([targetK[constrained_vert], g_dmat._K[constrained_vert], g_final._K[constrained_vert]]).T,fmt='%1.8f')

K_error = np.abs(g_final._K[constrained_vert]-targetK[constrained_vert])
bd_error = ( (fixed_coords-vert[fixed_vert])**2 )
l2 = np.sum( (vert[inedge[:,0]]-vert[inedge[:,1]])**2, axis=1 )
edge_error = np.abs(l2-edgelen**2)

print("edge^2 error: {}, boundary squared error: {}".format(np.sum(edge_error),np.sum(bd_error)))
print("curvature error (dmat-target): {}, (final-target): {}".format(np.abs(g_dmat._K[constrained_vert]-targetK[constrained_vert]).sum(), np.sum(K_error)))
#np.savetxt(os.path.join(args.outdir,"edge_final.csv"),np.hstack([inedge,dmat[inedge[:,0],inedge[:,1]][:,np.newaxis]]),delimiter=",",fmt="%i,%i,%f")

# graphs
sns.violinplot(y=edge_error, cut=0)
plt.savefig(os.path.join(args.outdir,"edge_error.png"))
plt.close()
sns.violinplot(y=bd_error, cut=0)
plt.savefig(os.path.join(args.outdir,"boundary_error.png"))
plt.close()

sns.violinplot(y=targetK, cut=0)
plt.savefig(os.path.join(args.outdir,"curvature_target.png"))
plt.close()
sns.violinplot(y=g_dmat._K[constrained_vert], cut=0)
plt.savefig(os.path.join(args.outdir,"curvature_dmat.png"))
plt.close()
sns.violinplot(y=g_final._K[constrained_vert], cut=0)
plt.savefig(os.path.join(args.outdir,"curvature_final.png"))
plt.close()
sns.violinplot(y=np.abs(g_dmat._K[constrained_vert]-targetK[constrained_vert]), cut=0)
plt.savefig(os.path.join(args.outdir,"error_dmat.png"))
plt.close()
sns.violinplot(y=np.abs(g_final._K[constrained_vert]-g_dmat._K[constrained_vert]), cut=0)
plt.savefig(os.path.join(args.outdir,"error_final_vs_dmat.png"))
plt.close()
sns.violinplot(y=np.abs(g_final._K[constrained_vert]-targetK[constrained_vert]), cut=0)
plt.savefig(os.path.join(args.outdir,"error_final.png"))
plt.close()

sns.violinplot(y=np.degrees(g_final.angle_array()), cut=0)
plt.savefig(os.path.join(args.outdir,"angles_final.png"))
plt.close()
