#!/usr/bin/env python
## pip install autograd
import argparse,os,time
import numpy as np
import seaborn as sns
from scipy.optimize import minimize,NonlinearConstraint,LinearConstraint,least_squares
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from ricci_flow import DiscreteRiemannianMetric, createMesh

def save_ply(vert,face,fname):
    el1 = PlyElement.describe(np.array([(x[0],x[1],x[2]) for x in vert],dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')]), 'vertex')
    el2 = PlyElement.describe(np.array([([x[0],x[1],x[2]], 0) for x in face],dtype=[('vertex_indices', 'i4', (3,)), ('red', 'u1')]), 'face')
    PlyData([el1,el2], text=True).write(fname)

def residual(x,edgelen2,inedge,cx,fixed_vert_idx,lambda_bdvert,fixed_beta):
    if fixed_beta:
        beta = 1
    else:
        beta = x[-1]
    v = np.reshape(x[:-1],(-1,3))
    l2 = np.sum( (v[inedge[:,0]]-v[inedge[:,1]])**2, axis=1 )
    loss = (l2-beta*edgelen2).ravel()
    loss = np.concatenate( [loss, np.sqrt(lambda_bdvert)* (v[fixed_vert_idx]-cx).ravel()] )
    return(loss)

def length_error(x,edgelen2,inedge):
    global n_iter
    beta = x[-1]
    v = np.reshape(x[:-1],(-1,3))
    l2 = np.sum( (v[inedge[:,0]]-v[inedge[:,1]])**2, axis=1 )
    loss = np.sum((l2-beta*edgelen2)**2 )
    if n_iter%20==0:
        print(n_iter,beta,loss)
    n_iter += 1
    return(loss)

def boundary_error(x,cx, fixed_vert_idx):
    v = np.reshape(x[:-1],(-1,3))[fixed_vert_idx]
    return np.sum( (v-cx)**2 )

# hard boundary constraints
def constraints(x):
    ## not implemented yet
    v=np.reshape(x,(-1,3))
    return np.array([x[0,0],x[0,1],x[N//3,1],x[2*N//3,1],x[N,0],x[N,1]])

def linear_combination_of_hessians(fun, argnum=0, *args, **kwargs):
    functionhessian = hessian(fun, argnum, *args, **kwargs)
    #not using wrap_nary_f because we need to do the docstring on our own
    def linear_combination_of_hessians(*funargs, **funkwargs):
        return np.tensordot(functionhessian(*funargs[:-1], **funkwargs), funargs[-1], axes=(0, 0))
    return linear_combination_of_hessians

#########################
parser = argparse.ArgumentParser(description='embedding of metric graphs')
parser.add_argument('input', help='Path to an input ply file')
parser.add_argument('--edge_length', '-el', default=None, help='Path to a csv specifying edge length')
parser.add_argument('--boundary_vertex', '-bv', default=None, help='Path to a csv specifying boundary position')
parser.add_argument('--method', '-m', default='trf',help='method for optimisation')
parser.add_argument('--outdir', '-o', default='result',help='Directory to output the result')
parser.add_argument('--lambda_bdvert', '-lv', type=float, default=1e-2, help="weight for boundary constraint")
parser.add_argument('--gtol', '-gt', type=float, default=1e-8, help="stopping criteria for gradient")
parser.add_argument('--verbose', '-v', action='store_true',help='print debug information')
parser.add_argument('--strict_boundary', '-sbd', action='store_true',help='strict boundary constraint')
parser.add_argument('--fixed_beta', '-fb', action='store_true',help='fix beta (distance scaling) as well')
args = parser.parse_args()

os.makedirs(args.outdir,exist_ok=True)

# Read mesh data
fn, ext = os.path.splitext(args.input)
fn = fn.rsplit('_', 1)[0]
if args.edge_length is None:
    args.edge_length =fn+"_edge.csv"
if args.boundary_vertex is None and args.lambda_bdvert>0:
    args.boundary_vertex =fn+"_boundary.csv"
if (ext==".ply"):
    plydata = PlyData.read(args.input)
    vert = np.vstack([plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z']]).astype(np.float64).T
    face = plydata['face']['vertex_indices']
    edgedat = np.loadtxt(args.edge_length,delimiter=",")
    inedge = edgedat[:,:2].astype(np.uint32)
    edgelen = edgedat[:,2]
    if args.boundary_vertex:
        bddat = np.loadtxt(args.boundary_vertex,delimiter=",")
        args.fixed_vert = bddat[:,0].astype(np.uint32)
        fixed_coords = bddat[:,1:]
    else:
        args.fixed_vert =np.array([0])
        fixed_coords = vert[args.fixed_vert]

else:  # old style input data format
    index_shift = 1
    vert = np.loadtxt(args.input)
    face = np.loadtxt(fn+"_face.txt").astype(np.uint16) -index_shift
    dmat = np.loadtxt(fn+"_edgeLength.txt")  # target distance mat
    inedge = np.loadtxt(fn+"_innerEdge.txt").astype(np.uint16) -index_shift
    args.fixed_vert = np.loadtxt(fn+"_bounVertex.txt").astype(np.uint16) -index_shift
    vert_init = np.loadtxt(fn+"_init.txt")
    fixed_coords = vert_init[args.fixed_vert]
    edgelen = dmat[inedge[:,0],inedge[:,1]]
    np.savetxt(fn+"_edge.csv", np.hstack([inedge,edgelen[:,np.newaxis]]),delimiter=",")
    np.savetxt(fn+"_boundary.csv", np.hstack([args.fixed_vert[:,np.newaxis],fixed_coords]),delimiter=",")

print("\nvertices {}, faces {}, fixed vertices {}".format(len(vert),len(face),len(args.fixed_vert)))

# initial scaling for dmat
#l1 = np.sum((vert[inedge[0,0]]-vert[inedge[0,1]])**2)
#edgelen2 = l1/(edgelen[0]**2) * (edgelen**2)
edgelen2 = edgelen**2

#%%
# initial point
x0 = np.concatenate([vert.flatten(),np.array([1.0])]) ## last entry is for scaling factor
n_iter=0

# optimise
print("optimising...")
start = time.time()
if args.method in ["lm","trf"]:
    res = least_squares(residual, x0, verbose=2, method=args.method, gtol=args.gtol, args=(edgelen2,inedge,fixed_coords,args.fixed_vert,args.lambda_bdvert,args.fixed_beta))
else:
    import autograd.numpy as np
    from autograd import grad, jacobian, hessian
    # jacobian and hessian by autograd
    print("computing gradient and hessian...")
    target = lambda x: length_error(x,edgelen2,inedge) + args.lambda_bdvert*boundary_error(x,fixed_coords,args.fixed_vert)
    jaco = jacobian(target)
    hess = hessian(target)
    xtol = 1e-5
    if args.strict_boundary: ## not implemented yet
        constraints_hess = linear_combination_of_hessians(constraints)
        constraints_jac = jacobian(constraints)
        hard_constraint = NonlinearConstraint(constraints, [0,0,1,-np.inf,1,0],[0,0,np.inf,-1,1,0], jac=constraints_jac, hess=constraints_hess)
        res = minimize(target, x0, method = 'trust-constr',
            options={'xtol': xtol, 'gtol': args.gtol, 'disp': True, 'verbose': 1}, jac = jaco, hess=hess, constraints=[hard_constraint])
    else:
        res = minimize(target, x0, method = 'trust-ncg',options={'gtol': args.gtol, 'disp': True}, jac = jaco, hess=hess)

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

#%% plot result
beta = res.x[-1]
vert2=np.reshape(res.x[:-1],(-1,3))

print("beta: {}, boundary squared error: {}".format(beta, (np.sum( (fixed_coords-vert2[args.fixed_vert])**2 ) )))

# output
bfn = os.path.basename(fn)
np.savetxt(os.path.join(args.outdir,bfn)+"_edge_scaled.csv",np.hstack([inedge,np.sqrt(beta*edgelen2[:,np.newaxis])]),delimiter=",",fmt="%i,%i,%f")
np.savetxt(os.path.join(args.outdir,bfn)+"_final.txt",vert2)
save_ply(vert2,face,os.path.join(args.outdir,bfn+"_final.ply"))

mesh = createMesh(vert2,[frozenset(x) for x in face])
g = DiscreteRiemannianMetric(mesh, mesh.lengths)
sns.violinplot(y=[g.curvature(i) for i in mesh.verts], cut=0)
plt.savefig(fn+"_curvature_final.png")
plt.close()
