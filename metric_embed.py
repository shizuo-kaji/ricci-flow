#!/usr/bin/env python
## pip install autograd
import argparse,os,time
import seaborn as sns
from scipy.optimize import minimize,NonlinearConstraint,LinearConstraint,least_squares
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from ricci_flow import DiscreteRiemannianMetric, TriangleMesh, save_ply

import numpy as np

def length_error(x,edgelen2,inedge,fixed_beta=False):
    #global n_iter
    if fixed_beta:
        beta = 1
    else:
        beta = x[-1]
    v = np.reshape(x[:-1],(-1,3))
    l2 = np.sum( (v[inedge[:,0]]-v[inedge[:,1]])**2, axis=1 )
    loss = (l2-beta*edgelen2).ravel()
#    if n_iter%20==0:
#        print(n_iter,beta,(loss**2).sum())
#    n_iter += 1
    return(loss)

def boundary_error(x,cx,fixed_vert_idx):
    v = np.reshape(x[:-1],(-1,3))
    return((v[fixed_vert_idx]-cx).ravel())

# quick-dirty convexity
def convexity_error(x,mesh):
    v = np.reshape(x[:-1],(-1,3))
    return(np.maximum(0,np.array([-v[i,2]+v[mesh.adj_vert[i],2].mean() for i in mesh.free_verts])))

#########################
parser = argparse.ArgumentParser(description='embedding of metric graphs')
parser.add_argument('input', help='Path to an input ply file')
parser.add_argument('--initial_point', '-ip', default=None, help='Path to a ply specifying the initial vertex positions')
parser.add_argument('--edge_length', '-el', default=None, help='Path to a csv specifying edge length')
parser.add_argument('--boundary_vertex', '-bv', default=None, help='Path to a csv specifying boundary position')
parser.add_argument('--inner_edge', '-ie', default=None, help='indices of inner edges')
parser.add_argument('--method', '-m', default='trf',help='method for optimisation')
parser.add_argument('--outdir', '-o', default='result',help='Directory to output the result')
parser.add_argument('--lambda_bdvert', '-lv', type=float, default=1.0, help="weight for boundary constraint")
parser.add_argument('--lambda_convex', '-lc', type=float, default=0, help="weight for convexity constraint")
parser.add_argument('--gtol', '-gt', type=float, default=1e-8, help="stopping criteria for gradient")
parser.add_argument('--verbose', '-v', type=int, default = 2)
parser.add_argument('--fixed_beta', '-fb', action='store_true',help='fix (not optimise) beta (distance scaling)')
args = parser.parse_args()

os.makedirs(args.outdir,exist_ok=True)

# Read mesh data
fn, ext = os.path.splitext(args.input)
pn, fn = os.path.split(fn)
fn = os.path.join(pn,os.path.basename(fn).rsplit('_', 1)[0])
if args.edge_length is None:
    args.edge_length =fn+"_edge.csv"
if args.boundary_vertex is None and args.lambda_bdvert>0:
    cfn = fn+"_boundary.csv"
    if os.path.isfile(cfn):
        args.boundary_vertex = cfn

#   
plydata = PlyData.read(args.input)
vert = np.vstack([plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z']]).astype(np.float64).T
face = plydata['face']['vertex_indices']
print("reading edge length from ", args.edge_length)
edgedat = np.loadtxt(args.edge_length,delimiter=",")
if args.inner_edge:
    index_shift=1
    print("reading inner edge indices from ", args.inner_edge)
    inedge = np.loadtxt(args.inner_edge).astype(np.uint16) -index_shift
    edgedict = {(i,j): l for i,j,l in zip(edgedat[:,0],edgedat[:,1],edgedat[:,2])}
    edgelen = np.array([edgedict[(e[0],e[1])] for e in inedge])
else:
    edgedat = np.array([ [i,j,l] for i,j,l in zip(edgedat[:,0],edgedat[:,1],edgedat[:,2]) if i<j ])
    inedge = edgedat[:,:2].astype(np.uint32)
    edgelen = edgedat[:,2]

fixed_coords = None
if args.boundary_vertex:
    print("reading boundary data from ", args.boundary_vertex)
    bddat = np.loadtxt(args.boundary_vertex,delimiter=",")
    if len(bddat)>0:
        args.fixed_vert = bddat[:,0].astype(np.uint32)
        fixed_coords = bddat[:,1:]
if fixed_coords is None:
    args.lambda_bdvert = 0
    args.fixed_vert =np.array([0])
    fixed_coords = vert[args.fixed_vert]

print("\nvertices {}, faces {}, fixed vertices {}".format(len(vert),len(face),len(args.fixed_vert)))
if len(args.fixed_vert) < 2:
    args.fixed_beta = True # otherwise, everything shrinks to a point

# initial scaling for dmat
#l1 = np.sum((vert[inedge[0,0]]-vert[inedge[0,1]])**2)
#edgelen2 = l1/(edgelen[0]**2) * (edgelen**2)
edgelen2 = edgelen**2

#%%
# initial point
# vert[:,2] = np.abs(2*vert[:,2]) ## force positive z for the initial point
if args.initial_point is not None:
    plydata = PlyData.read(args.initial_point)
    x0 = np.concatenate([np.vstack([plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z']]).astype(np.float64).T.ravel(),np.array([1.0])])
else:
    x0 = np.concatenate([vert.flatten(),np.array([1.0])]) ## last entry is for scaling factor

# optimise
n_iter=0
start = time.time()
if args.lambda_convex>0:
    mesh = TriangleMesh(vert,face)
else:
    mesh = None

wb = np.sqrt(args.lambda_bdvert*len(edgelen2)/max(1,len(args.fixed_vert)))
wc = np.sqrt(args.lambda_convex*len(edgelen2)/max(1,len(vert)))
if args.method in ["lm","trf"]:
    if args.lambda_convex > 0:
        target = lambda x: np.concatenate([length_error(x,edgelen2,inedge,args.fixed_beta), wb*boundary_error(x,fixed_coords,args.fixed_vert) , wc*convexity_error(x,mesh)])
    else:
        target = lambda x: np.concatenate([length_error(x,edgelen2,inedge,args.fixed_beta), wb*boundary_error(x,fixed_coords,args.fixed_vert)])
#    bd = (np.full(len(x0),-np.inf), np.full(len(x0),np.inf))
#    bd[0][2::3] = 0  ## set lower bound of z
#    res = least_squares(target, x0, bounds=bd, verbose=2, method=args.method, gtol=args.gtol)
    #res = least_squares(target, x0, jac=jacobian(target), verbose=2, method=args.method, gtol=args.gtol)
    res = least_squares(target, x0, verbose=args.verbose, method=args.method, gtol=args.gtol)
else:
    import autograd.numpy as np
    from autograd import grad, jacobian, hessian
    # jacobian and hessian by autograd
    target = lambda x: np.sum(length_error(x,edgelen2,inedge)**2) + wb**2*np.sum(boundary_error(x,fixed_coords,args.fixed_vert)**2)
    if args.method in ['CG','BFGS']:
        res = minimize(target, x0, method = args.method,options={'gtol': args.gtol, 'disp': True}, jac = jacobian(target))
    else:
        res = minimize(target, x0, method = args.method,options={'gtol': args.gtol, 'disp': True}, jac = jacobian(target), hess=hessian(target))

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

#%% plot result
beta = res.x[-1]
vert2=np.reshape(res.x[:-1],(-1,3))
#print(vert2[:,2].min())

print("beta: {}, cost: {}, boundary MAE: {}".format(beta, res.fun.sum(), (np.sqrt(np.sum( (fixed_coords-vert2[args.fixed_vert])**2, axis=1 )) ).mean()  ))

# output
bfn = os.path.basename(fn)
bfn = os.path.join(args.outdir,bfn)
np.savetxt(bfn+"_edge_scaled.csv",np.hstack([inedge,np.sqrt(beta*edgelen2[:,np.newaxis])]),delimiter=",",fmt="%i,%i,%f")
#np.savetxt(bfn+"_final.txt",vert2)
save_ply(vert2,face,bfn+"_final.ply")

mesh = TriangleMesh(vert2,face)
g = DiscreteRiemannianMetric(mesh, mesh.lengths)
sns.violinplot(y=[g.curvature(i) for i in mesh.verts], cut=0)
plt.savefig(bfn+"_curvature_final.png")
plt.close()
