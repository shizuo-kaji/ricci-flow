#!/usr/bin/env python
## pip install autograd
import argparse,os,time
import seaborn as sns
from scipy import sparse
from scipy.optimize import minimize,NonlinearConstraint,LinearConstraint,least_squares
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from ricci_flow import DiscreteRiemannianMetric, TriangleMesh, save_ply
import subprocess,sys

import numpy as np

def length_error(x,edgelen2,inedge,edgeweight,fixed_beta=False):
    #global n_iter
    if fixed_beta:
        beta = 1
    else:
        beta = x[-1]
    v = np.reshape(x[:-1],(-1,3))
    l2 = np.sum( (v[inedge[:,0]]-v[inedge[:,1]])**2, axis=1 )
    loss = edgeweight*(l2-beta*edgelen2)
#    if n_iter%20==0:
#        print(n_iter,beta,(loss**2).sum())
#    n_iter += 1
    return(loss)

def grad_length_error(x,edgelen2,inedge,edgeweight,fixed_beta=False):
    v = np.reshape(x[:-1],(-1,3))
    Hm = sparse.dok_matrix((len(inedge),len(x)))
    for i,e in enumerate(inedge):
        # 2*(v[inedge[:,0]]-v[inedge[:,1]])
        Hm[i, 3*e[0]:(3*e[0]+3)] =  edgeweight[i]*2*(v[e[0]]-v[e[1]])
        Hm[i, 3*e[1]:(3*e[1]+3)] = -edgeweight[i]*2*(v[e[0]]-v[e[1]])
        # for j in range(3):
        #     Hm[i, 3*e[0]+j] =  2*(v[e[0],j]-v[e[1],j])
        #     Hm[i, 3*e[1]+j] = -2*(v[e[0],j]-v[e[1],j])
        Hm[i,-1] = -edgeweight[i]*edgelen2[i]
    return Hm.tocsr()

# dense version
def dgrad_length_error(x,edgelen2,inedge,edgeweight,fixed_beta=False):
    v = np.reshape(x[:-1],(-1,3))
    Hm = np.zeros((len(inedge),len(x)))
    for i,e in enumerate(inedge):
        # 2*(v[inedge[:,0]]-v[inedge[:,1]])
        Hm[i, 3*e[0]:(3*e[0]+3)] =  edgeweight[i]*2*(v[e[0]]-v[e[1]])
        Hm[i, 3*e[1]:(3*e[1]+3)] = -edgeweight[i]*2*(v[e[0]]-v[e[1]])
        Hm[i,-1] = -edgeweight[i]*edgelen2[i]
    return Hm

def boundary_error(x,cx,fixed_vert_idx,vertweight):
    v = np.reshape(x[:-1],(-1,3))
    return((vertweight[:,np.newaxis]*(v[fixed_vert_idx]-cx)).ravel())

def grad_boundary_error(x,cx,fixed_vert_idx,vertweight):
    Hm = sparse.dok_matrix((3*len(cx),len(x)))
    for i,k in enumerate(fixed_vert_idx):
        for j in range(3):
            Hm[3*i+j, 3*k+j] = vertweight[i]
    return Hm.tocsr()

# dense version
def dgrad_boundary_error(x,cx,fixed_vert_idx,vertweight):
    Hm = np.zeros((3*len(cx),len(x)))
    for i,k in enumerate(fixed_vert_idx):
        for j in range(3):
            Hm[3*i+j, 3*k+j] = vertweight[i]
    return Hm

# quick-dirty convexity
def convexity_error(x,mesh):
    v = np.reshape(x[:-1],(-1,3))
    #return(np.maximum(0,np.array([-v[i,2]+v[mesh.adj_vert[i],2].mean() for i in mesh.free_verts])))
    u = np.array([-v[i,2]+v[mesh.adj_vert[i],2].mean() for i in mesh.free_verts])
    return(u/(1+np.exp(-u)))

def grad_convexity_error(x,mesh):
    # Swish: f'(x) = β f(βx) + σ(βx)(1 – β f(βx))
    v = np.reshape(x[:-1],(-1,3))
    u = np.array([-v[i,2]+v[mesh.adj_vert[i],2].mean() for i in mesh.free_verts])
    f = u/(1+np.exp(-u))
    df = f + (1-f)/(1+np.exp(-u))
    Hm = sparse.dok_matrix((len(mesh.free_verts),len(x)))
    for i,k in enumerate(mesh.free_verts):
        Hm[i, 3*i+2] = -df[i]
        deg = len(mesh.adj_vert[k])
        for j in mesh.adj_vert[k]:
            Hm[i, 3*j+2] = df[i]/deg
    return Hm.tocsr()

# dense version
def dgrad_convexity_error(x,mesh):
    v = np.reshape(x[:-1],(-1,3))
    u = np.array([-v[i,2]+v[mesh.adj_vert[i],2].mean() for i in mesh.free_verts])
    f = u/(1+np.exp(-u))
    df = f + (1-f)/(1+np.exp(-u))
    Hm = np.zeros((len(mesh.free_verts),len(x)))
    for i,k in enumerate(mesh.free_verts):
        Hm[i, 3*i+2] = -df[i]
        deg = len(mesh.adj_vert[k])
        for j in mesh.adj_vert[k]:
            Hm[i, 3*j+2] = df[i]/deg
    return Hm

#########################
parser = argparse.ArgumentParser(description='embedding of metric graphs')
parser.add_argument('input', help='Path to an input ply file')
parser.add_argument('--initial_point', '-ip', default=None, help='Path to a ply specifying the initial vertex positions')
parser.add_argument('--edge_length', '-el', default=None, help='Path to a csv specifying edge length')
parser.add_argument('--boundary_vertex', '-bv', default=None, help='Path to a csv specifying boundary position')
parser.add_argument('--optimizer', '-op', default='trf',help='method for optimisation')
parser.add_argument('--outdir', '-o', default='result',help='Directory to output the result')
parser.add_argument('--lambda_bdvert', '-lv', type=float, default=0.01, help="weight for boundary position constraint")
parser.add_argument('--lambda_bdedge', '-le', type=float, default=-1, help="weight for boundary edge length constraint")
parser.add_argument('--lambda_convex', '-lc', type=float, default=0, help="weight for convexity constraint")
parser.add_argument('--norm_order', '-no', type=int, default=2, help="norm order for autograd optimisers")
parser.add_argument('--gtol', '-gt', type=float, default=1e-8, help="stopping criteria for gradient")
parser.add_argument('--verbose', '-v', type=int, default = 2)
parser.add_argument('--fixed_beta', '-fb', action='store_true',help='fix (not optimise) beta (distance scaling)')
parser.add_argument('--evaluation', '-e', action='store_true',help='perform evaluation as well')
parser.add_argument('--coloured_ply', '-cp', type=float, nargs=2, help='vertices in out PLY are coloured by their curvature')
parser.add_argument('--jitter', '-j', type=float, default=0, help='jitter initial coordinates to make them in general position.')
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
mesh = None # loaded later if necessary
print("reading edge length from ", args.edge_length)
edgedat = np.loadtxt(args.edge_length,delimiter=",")
if edgedat.shape[1]==4:  ## with weight information
    edgedat = np.array([ [i,j,l,w] for i,j,l,w in zip(edgedat[:,0],edgedat[:,1],edgedat[:,2],edgedat[:,3]) if i<j ])
    edgeweight = edgedat[:,3]
else:
    edgedat = np.array([ [i,j,l] for i,j,l in zip(edgedat[:,0],edgedat[:,1],edgedat[:,2]) if i<j ])
    edgeweight = np.ones(len(edgedat))
inedge = edgedat[:,:2].astype(np.uint32)
edgelen = edgedat[:,2]

# edge weight correction
if args.lambda_bdedge>-1:
    mesh = TriangleMesh(vert,face)
    for k,(i,j) in enumerate(inedge):
        if frozenset({i,j}) in mesh.b_edges:
            edgeweight[k]=args.lambda_bdedge
#    print(len(mesh.b_edges),sum(edgeweight==args.lambda_bdedge))    

# boundary vertex coordinates
fixed_coords = None
if args.boundary_vertex:
    print("reading boundary data from ", args.boundary_vertex)
    bddat = np.loadtxt(args.boundary_vertex,delimiter=",")
    if len(bddat)>0:
        if bddat.shape[1]==5:
            vertweight = bddat[:,4]
        else:
            vertweight = np.ones(len(bddat))
        args.fixed_vert = bddat[vertweight>0,0].astype(np.uint32)
        fixed_coords = bddat[vertweight>0,1:4]
if fixed_coords is None:
    args.lambda_bdvert = 0
    args.fixed_vert =np.array([0])
    fixed_coords = vert[args.fixed_vert].copy()
    vertweight = np.ones(len(fixed_coords))

if args.jitter>0:
    mx, Mx = np.min(vert[:,0]), np.max(vert[:,0])
    my, My = np.min(vert[:,1]), np.max(vert[:,1])
    cx, cy = (Mx+mx)/2, (My+my)/2
    diam = (Mx-cx)**2 + (My-cy)**2
    vert[:,2] += args.jitter*(Mx-mx)*np.sqrt(diam-(vert[:,0]-cx)**2-(vert[:,1]-cy)**2)

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
if args.lambda_convex>0 and mesh is None:
    mesh = TriangleMesh(vert,face)

wb = np.sqrt(args.lambda_bdvert*len(edgelen2)/max(1,len(args.fixed_vert)))
wc = np.sqrt(args.lambda_convex*len(edgelen2)/max(1,len(vert)))
if args.optimizer in ["lm","trf"]:
    if args.lambda_convex > 0 and len(fixed_coords)>0 and wb>0:
        target = lambda x: np.concatenate([length_error(x,edgelen2,inedge,edgeweight,args.fixed_beta), wb*boundary_error(x,fixed_coords,args.fixed_vert,vertweight) , wc*convexity_error(x,mesh)])
        if args.optimizer == "lm":
            jac = lambda x: np.vstack([dgrad_length_error(x,edgelen2,inedge,edgeweight,args.fixed_beta), wb*dgrad_boundary_error(x,fixed_coords,args.fixed_vert,vertweight), wc*dgrad_convexity_error(x,mesh)]).tolist()
        else:
            jac = lambda x: sparse.vstack([grad_length_error(x,edgelen2,inedge,edgeweight,args.fixed_beta), wb*grad_boundary_error(x,fixed_coords,args.fixed_vert,vertweight), wc*grad_convexity_error(x,mesh)])
    elif len(fixed_coords)>0 and wb>0:
        target = lambda x: np.concatenate([length_error(x,edgelen2,inedge,edgeweight,args.fixed_beta), wb*boundary_error(x,fixed_coords,args.fixed_vert,vertweight)])
        if args.optimizer == "lm":
            jac = lambda x: np.vstack([dgrad_length_error(x,edgelen2,inedge,edgeweight,args.fixed_beta), wb*dgrad_boundary_error(x,fixed_coords,args.fixed_vert,vertweight)]).tolist()
        else:
            jac = lambda x: sparse.vstack([grad_length_error(x,edgelen2,inedge,edgeweight,args.fixed_beta), wb*grad_boundary_error(x,fixed_coords,args.fixed_vert,vertweight)])
    else:
        target = lambda x: length_error(x,edgelen2,inedge,edgeweight,args.fixed_beta)
        if args.optimizer == "lm":
            jac = lambda x: np.vstack([dgrad_length_error(x,edgelen2,inedge,edgeweight,args.fixed_beta)]).tolist()
        else:
            jac = lambda x: grad_length_error(x,edgelen2,inedge,edgeweight,args.fixed_beta)

#    jac = '2-point'
# box constraint
#    bd = (np.full(len(x0),-np.inf), np.full(len(x0),np.inf))
#    bd[0][2::3] = 0  ## set lower bound of z
#    res = least_squares(target, x0, bounds=bd, verbose=2, method=args.optimizer, gtol=args.gtol)
    res = least_squares(target, x0, jac=jac, verbose=args.verbose, method=args.optimizer, gtol=args.gtol)
else:
    import autograd.numpy as np
    from autograd import grad, jacobian, hessian
    ord = args.norm_order
    if wc>0:
        target = lambda x: np.linalg.norm(length_error(x,edgelen2,inedge,edgeweight),ord=ord) + wb**2*np.linalg.norm(boundary_error(x,fixed_coords,args.fixed_vert,vertweight),ord=ord) + wc**2*np.linalg.norm(convexity_error(x,mesh),ord=ord)
    else:
        target = lambda x: np.linalg.norm(length_error(x,edgelen2,inedge,edgeweight),ord=ord) + wb**2*np.linalg.norm(boundary_error(x,fixed_coords,args.fixed_vert,vertweight),ord=ord)
    if ord == 1:
        jac = None
    else:
        jac = jacobian(target)
    if args.optimizer in ['CG','BFGS']:
        hess = None
    else:
        hess = hessian(target)
    print("autograd")
    res = minimize(target, x0, method = args.optimizer,options={'gtol': args.gtol, 'disp': True}, jac = jac, hess=hess)

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

#%% plot result
if args.fixed_beta:
    beta = 1
else:
    beta = res.x[-1]
vert2=np.reshape(res.x[:-1],(-1,3))
#print(vert2[:,2].min())

if args.optimizer in ["lm","trf"]:
    print("beta: {}, cost: {}, boundary MAE: {}".format(beta, np.abs(res.cost).sum(), (np.sqrt(np.sum( (fixed_coords-vert2[args.fixed_vert])**2, axis=1 )) ).mean()  ))

# output
bfn = os.path.basename(fn)
bfn = os.path.join(args.outdir,bfn)
np.savetxt(bfn+"_edge_scaled.csv",np.hstack([inedge,np.sqrt(beta*edgelen2[:,np.newaxis])]),delimiter=",",fmt="%i,%i,%f")
#np.savetxt(bfn+"_final.txt",vert2)
if args.coloured_ply is not None:
    mesh_final = TriangleMesh(vert2,face)
    g_final = DiscreteRiemannianMetric(mesh_final, mesh_final.lengths)
    if args.coloured_ply[0]>2*np.pi:
        vmin = np.min(g_final._K)
    else:
        vmin = args.coloured_ply[0]
    if args.coloured_ply[1]>2*np.pi:
        vmax = np.max(g_final._K)
    else:
        vmax = args.coloured_ply[1]
    colour = np.array(255*plt.cm.bwr((g_final._K-vmin)/(vmax-vmin)), dtype=np.uint8)
else:
    colour = None
save_ply(vert2,face,bfn+"_final.ply",colour)

if args.evaluation:
    dn = os.path.dirname(__file__)
    cmd = "python {} {} -o {}".format(os.path.join(dn,"evaluation.py"),fn+"_final.ply", args.outdir)
    print("\n",cmd)
    subprocess.call(cmd, shell=True)
    