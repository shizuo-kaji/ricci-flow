#!/usr/bin/env python
import argparse,os,time
import seaborn as sns
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import numpy as np
from ricci_flow import DiscreteRiemannianMetric, TriangleMesh

#########################
parser = argparse.ArgumentParser(description='embedding of metric graphs')
parser.add_argument('input', help='final mesh')
parser.add_argument('--boundary_vertex', '-bv', default=None, help='Path to a csv specifying boundary position')
parser.add_argument('--initial_mesh', '-im', default=None, help='initial mesh')
parser.add_argument('--edge_length', '-el', default=None, help='Path to a csv specifying edge length')
parser.add_argument('--index_shift', type=int, default=1, help="vertex indices start at")
parser.add_argument('--outdir', '-o', default='result',help='Directory to output the result')
parser.add_argument('--target_curvature', '-K', default=None, type=str, help='file specifying target gaussian curvature')
parser.add_argument('--target_curvature_interior', '-Ki', default=0.01181102, type=float, help='target gaussian curvature value')
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
if args.initial_mesh is None:
    cfn = fn+".ply"
    if os.path.isfile(cfn):
        args.initial_mesh = cfn
if args.target_curvature is None:
    cfn = fn+"_targetCurvature.txt"
    if os.path.isfile(cfn):
        args.target_curvature = cfn

#print(args)
plydata = PlyData.read(args.input)
print("reading final mesh from ", args.input)
vert = np.vstack([plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z']]).T
#    vert2 = np.loadtxt(fn+"_init.txt")
#    print(np.abs(vert-vert2).sum())
face = plydata['face']['vertex_indices']

edgedat = np.loadtxt(args.edge_length,delimiter=",")
print("reading edge lengths from ", args.edge_length)
edgedat = np.array([ [i,j,l] for i,j,l in zip(edgedat[:,0],edgedat[:,1],edgedat[:,2]) if i<j ])
edgedict = {frozenset([i,j]): l for i,j,l in zip(edgedat[:,0],edgedat[:,1],edgedat[:,2])}
inedge = edgedat[:,:2].astype(np.uint32)
edgelen = edgedat[:,2]
#dmat = np.sum((np.expand_dims(vert,axis=0) - np.expand_dims(vert,axis=1))**2,axis=2)
#dmat[inedge[:,0],inedge[:,1]] = edgelen
#dmat[inedge[:,1],inedge[:,0]] = edgelen
#    np.savetxt("dmat.txt",dmat)

fixed_coords = None
if args.boundary_vertex:
    print("reading boundary data from ", args.boundary_vertex)
    bddat = np.loadtxt(args.boundary_vertex,delimiter=",")
    if len(bddat)>0:
        fixed_vert = bddat[:,0].astype(np.uint32)
        fixed_coords = bddat[:,1:]
if fixed_coords is None:
    fixed_vert =np.array([0])
    fixed_coords = vert[fixed_vert]

# set target curvature
if args.target_curvature:
    print("reading target curvature from ", args.target_curvature)
    targetK = np.loadtxt(args.target_curvature)
    KspecifiedV = np.where(targetK <= 2*np.pi)[0]
else:
    targetK = np.full(len(vert),args.target_curvature_interior)
    KspecifiedV = range(len(vert))

#
args.free_vert = list(set(range(len(vert))) - set(fixed_vert))

print("\nvertices {}, faces {}, fixed verts {}, K-specified verts {}".format(len(vert),len(face),len(fixed_vert),len(KspecifiedV)))

mesh_final = TriangleMesh(vert,face)
g_final = DiscreteRiemannianMetric(mesh_final, mesh_final.lengths)
g_dmat = DiscreteRiemannianMetric(mesh_final, edgedict)

## conformality
if args.initial_mesh is not None:
    print("reading initial mesh from ", args.initial_mesh)
    plydata_init = PlyData.read(args.initial_mesh)
    vert_init = np.vstack([plydata_init['vertex']['x'],plydata_init['vertex']['y'],plydata_init['vertex']['z']]).T
    face_init = plydata_init['face']['vertex_indices']
    mesh_init = TriangleMesh(vert_init,face_init)
    g_init = DiscreteRiemannianMetric(mesh_init, mesh_init.lengths)
    angles_init = np.degrees(g_init.angle_array(mesh_init.b_verts))
    angles_final = np.degrees(g_final.angle_array(mesh_init.b_verts))
    angles_dmat = np.degrees(g_dmat.angle_array(mesh_init.b_verts))
    np.savetxt(os.path.join(args.outdir,"angles_init_dmat_final.csv"),np.vstack([angles_init,angles_dmat,angles_final]).T,fmt='%1.8f',delimiter=",")
    print("angle MAE: (init vs final) {}, (init vs dmat) {}".format( np.abs(angles_init-angles_final).mean(),  np.abs(angles_init-angles_dmat).mean() ))
    plt.violinplot([angles_init,angles_final])
    plt.savefig(os.path.join(args.outdir,"angles_init_final.png"))
    plt.close()
    sns.violinplot(y=g_init._K[KspecifiedV], cut=0)
    plt.savefig(os.path.join(args.outdir,"curvature_init.png"))
    plt.close()
    np.savetxt(os.path.join(args.outdir,"curvatures_init_target_dmat_final.csv"),np.vstack([g_init._K[KspecifiedV], targetK[KspecifiedV], g_dmat._K[KspecifiedV], g_final._K[KspecifiedV]]).T,fmt='%1.8f',delimiter=",")


# %%

K_error = np.abs(g_final._K[KspecifiedV]-targetK[KspecifiedV])
bd_error = (np.sqrt(np.sum( (fixed_coords-vert[fixed_vert])**2, axis=1 )) ) 
l2 = np.sum( (vert[inedge[:,0]]-vert[inedge[:,1]])**2, axis=1 )
edge_error = np.abs(l2-edgelen**2)

print("edge length MAE (dmat vs final): {} \nboundary MAE: {}".format(np.mean(edge_error),np.mean(bd_error)))
print("curvature MAE (final vs target): {}, (dmat vs target): {}".format(np.mean(K_error),np.abs(g_dmat._K[KspecifiedV]-targetK[KspecifiedV]).mean()))
#np.savetxt(os.path.join(args.outdir,"edge_final.csv"),np.hstack([inedge,dmat[inedge[:,0],inedge[:,1]][:,np.newaxis]]),delimiter=",",fmt="%i,%i,%f")

# graphs
sns.violinplot(y=edge_error, cut=0)
plt.savefig(os.path.join(args.outdir,"error_edge_length.png"))
plt.close()
sns.violinplot(y=bd_error, cut=0)
plt.savefig(os.path.join(args.outdir,"error_boundary.png"))
plt.close()

plt.violinplot([targetK[KspecifiedV],g_dmat._K[KspecifiedV],g_final._K[KspecifiedV]])
plt.savefig(os.path.join(args.outdir,"curvature_target_dmat_final.png"))
plt.close()
sns.violinplot(y=np.abs(g_dmat._K[KspecifiedV]-targetK[KspecifiedV]), cut=0)
plt.savefig(os.path.join(args.outdir,"error_curvature_dmat.png"))
plt.close()
sns.violinplot(y=np.abs(g_final._K[KspecifiedV]-g_dmat._K[KspecifiedV]), cut=0)
plt.savefig(os.path.join(args.outdir,"error_curvature_final_vs_dmat.png"))
plt.close()
sns.violinplot(y=np.abs(g_final._K[KspecifiedV]-targetK[KspecifiedV]), cut=0)
plt.savefig(os.path.join(args.outdir,"error_curvature_final.png"))
plt.close()
