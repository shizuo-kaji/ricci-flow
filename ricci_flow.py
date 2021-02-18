## original by Harrison Chapman https://github.com/hchapman/ricci-flow
## bug fix/modified by S. Kaji (Dec. 2020)

import numpy as np
from scipy import sparse
from numpy.linalg import norm, inv, lstsq, solve
from itertools import combinations, product
import functools
from numpy import dot
from scipy.sparse import linalg
from plyfile import PlyData, PlyElement
import argparse,os,time
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize,NonlinearConstraint,LinearConstraint,least_squares
import subprocess

def isfloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def ricciEnergy(u, eta, targetK, KspecifiedV, boundary_u, mesh, alpha=0):
    g = np.exp(u)
    K = np.full(len(u),2*np.pi)
    K[mesh.b_verts] = np.pi
    for i,j,k in mesh.faces:
        Li = (2*g[j]*g[k]*eta[j,k] + g[j]**2 + g[k]**2)
        Lj = (2*g[k]*g[i]*eta[k,i] + g[k]**2 + g[i]**2)
        Lk = (2*g[i]*g[j]*eta[i,j] + g[i]**2 + g[j]**2)
        K[i] -= np.arccos(np.clip((Lj+Lk-Li)/(2*np.sqrt(Lj*Lk)),-1.0,1.0))
        K[j] -= np.arccos(np.clip((Lk+Li-Lj)/(2*np.sqrt(Lk*Li)),-1.0,1.0))
        K[k] -= np.arccos(np.clip((Li+Lj-Lk)/(2*np.sqrt(Li*Lj)),-1.0,1.0))

    return(np.concatenate( [K[KspecifiedV]-targetK[KspecifiedV], np.sqrt(alpha) * (u[mesh.b_verts]-boundary_u)] ))

class IdentityDictMap(object):
    def __init__(self, output=1, domain=None):
        self._o = output
        self._domain = domain
    def __getitem__(self, i):
        if self._domain and i not in self._domain:
            raise KeyError()
        return self._o

def save_ply(vert,face,fname):
    el1 = PlyElement.describe(np.array([(x[0],x[1],x[2]) for x in vert],dtype=[('x', 'f8'), ('y', 'f8'),('z', 'f8')]), 'vertex')
    el2 = PlyElement.describe(np.array([([x[0],x[1],x[2]], 0) for x in face],dtype=[('vertex_indices', 'i4', (3,)), ('red', 'u1')]), 'face')
    PlyData([el1,el2], text=True).write(fname)

def partition_face(face):
    i,j,k = face
    yield i, (j,k)
    yield j, (k,i)
    yield k, (i,j)

def triangulate(poly):
    if len(poly) == 3:
        yield poly
    else:
        e0 = poly[0]
        for e1, e2 in zip(poly[1:], poly[2:]):
            yield (e0, e1, e2)

def read_obj_file(f):
    verts = []
    normals = []
    faces = []
    for line in f:
        tok = line.split()
        if not tok:
            continue
        elif tok[0] == "#":
            continue
        elif tok[0] == "o":
            print("Reading %s"%tok[1])
        elif tok[0] == "v":
            verts.append(np.array([float(x) for x in tok[1:]]))
        elif tok[0] == "vn":
            normals.append(np.array([float(x) for x in tok[1:]]))
        elif tok[0] == "f":
            # Throw out normal info for now...
            poly = [int(vstr.split("/")[0])-1 for vstr in tok[1:]]
            for face in triangulate(poly):
                faces.append(face)

    verts = np.array(verts)
    return(verts,faces)


class TriangleMesh(object):
    def __init__(self, vert_coords, faces, edges=None, lengths=None, bg_geom="euclidean"):
        self.faces = [frozenset(f) for f in faces]
        if edges is None:
            edges = []
            lengths = dict()
            boundaries = []
            maxnorm = max(np.linalg.norm(v) for v in vert_coords)
            vert_coords = vert_coords/maxnorm/2
            for face in self.faces:
                for edge in combinations(face, 2):
                    elen = norm(vert_coords[edge[0]] - vert_coords[edge[1]])
                    edge = frozenset(edge)
                    if edge not in edges:
                        edges.append(edge)
                        lengths[edge] = elen
                        boundaries.append(edge)
                    else:
                        boundaries.remove(edge)
            if boundaries:
                b_verts = sorted(list(reduce(lambda A,B: A.union(B), boundaries)))
            else:
                b_verts = []

        self.edges = edges
        self.lengths = lengths
        self.verts = list(range(len(vert_coords))) ## id
        self.b_verts = b_verts ## id
        self.free_verts = list(set(self.verts)-set(b_verts))
        self.bg_geom = bg_geom ## not used yet
        self.adj_faces = [[face for face in self.faces if vert in face] for vert in self.verts]

        # # neighbouring vertices
        # self.N = []
        # for v in self.verts:
        #     F = [face for face in faces if v in face]
        #     print(v,F)
        #     f = F.pop()
        #     i = f.index(v)
        #     L = [f[(i+1) % 3],f[(i+2) % 3]]
        #     print(L)
        #     while(F):
        #         for i in range(len(F)):
        #             if L[-1] in F[i]:
        #                 f = F.pop(i)
        #                 i = f.index(L[-1])
        #                 L.append(f[(i+1) % 3])
        #                 break
        #         for i in range(len(F)):
        #             if L[0] in F[i]:
        #                 f = F.pop(i)
        #                 i = f.index(L[0])
        #                 L.append(f[(i-1) % 3])
        #                 break
        #     print(v, L)
        #     self.N.append(L)

    def adjacent_edges(self, vert):
        return [edge for edge in self.edges if vert in edge]

    def min_valence(self):
        return min([len([face for face in self.faces if vert in face]) for vert in self.verts])

    def chi(self):
        return len(self.verts) - len(self.edges) + len(self.faces)

class DiscreteRiemannianMetric(object):
    def __init__(self, mesh, length_map):
        self._n = len(mesh.verts)
        self._mesh = mesh

        # Lengths l_{ij}
#        self._lmap = sparse.lil_matrix((self._n, self._n))
        self._l = dict()
#        self._l = sparse.dok_matrix((self._n, self._n))

        # Angles \theta_i^{jk}
        self._theta = dict()

        # Curvatures K_i
        self._K = None

        # symmetrise edge length
        for (i,j), l in length_map.items():
            self._l[i,j] = l
            self._l[j,i] = l

        self.update()

    def update(self):
        # Set angles using law of cosines
        for face in self._mesh.faces:
            for i,(j,k) in partition_face(face):
                theta = self.compute_angle(face,i)
                self._theta[(i,j,k)] = theta
                self._theta[(i,k,j)] = theta

        # Set curvatures
        self._K = self.curvature_array()

    def is_ok(self): ## check triangle inequality
        for face in self._mesh.faces:
            edges = [frozenset(e) for e in combinations(face,2)]
            k = len(edges)
            for i in range(k):
                if self.length(edges[(i+2)%k]) + self.length(edges[(i+1)%k]) < self.length(edges[i]):
                    return False
        return True

    def abc_for_vert(self, face, vert): # enumerate edge lengths of a face
        assert(vert in face)
        other_v = list(face - set([vert]))
        edge_a = [vert, other_v[0]]
        edge_b = [vert, other_v[1]]
        edge_c = other_v
        return [self.length(e_i) for e_i in (edge_a,edge_b,edge_c)]

    def angle(self, face, vert):
        j,k = face - set([vert])
        return self._theta[(vert,j,k)]

    def angle_array(self):
        U = []
        for f in self._mesh.faces:
            for v in f:
                U.append(self.angle(f,v))
        return(np.array(U))

    def compute_angle(self, face, vert):
        a,b,c = self.abc_for_vert(face,vert)
        try:   ## this is faster than np.clip or np.nan_to_num
            return np.arccos((a**2 + b**2 - c**2)/(2.0*a*b))
        except FloatingPointError:
            return 0.0001
 #       return(np.arccos( np.clip( (a**2 + b**2 - c**2)/(2.0*a*b), -1.0, 1.0)) )

    def curvature(self, vert):
        if vert in self._mesh.b_verts:
            return np.pi - sum([self.angle(face, vert) for face in self._mesh.adj_faces[vert]])
        else:
            return 2*np.pi - sum([self.angle(face, vert) for face in self._mesh.adj_faces[vert]])

    def curvature_array(self):
        K = np.zeros(len(self._mesh.verts))
        for v_i in self._mesh.verts:
            K[v_i] = self.curvature(v_i)
        return K

    def length(self, edge):
        i,j = edge
        return self._l[i,j]

    def total_curvature(self):
        return sum([self.curvature(vert) for vert in self._mesh.verts])

    def face_area(self, face):
        i,j,k = face
        gamma = self._theta[(i,j,k)]
        a,b = self.length((i,j)), self.length((i,k))
        return .5*a*b*np.sin(gamma)

    def area(self):
        return sum([self.face_area(face) for face in self._mesh.faces])

    def gb_chi(self):
        EPSILON_DICT = {
            'euclidean':   0,
            'spherical':   1,
            'hyperbolic': -1,
        }
        return (self.total_curvature() + EPSILON_DICT[self._mesh.bg_geom]*self.area())/(np.pi*2)

    def conf_factor(self, gamma):
        return np.log(gamma)

    # for hessian computation
    def _Theta(self, face):
        i,j,k = face
        theta = functools.partial(self.angle, face)
        return np.cos(np.array(
            ((np.pi,            theta(k), theta(j)),
             (theta(k), np.pi,            theta(i)),
             (theta(j), theta(i), np.pi           )
         )))

    def _D(self, tau, i,j,k):
        return np.array(
            ((0,                tau(i,j,k), tau(i,k,j)),
             (tau(j,i,k), 0,                tau(j,k,i)),
             (tau(k,i,j), tau(k,j,i), 0               )
         ))


class CirclePackingMetric(DiscreteRiemannianMetric):
    def __init__(self, mesh, radius_map, struct_coeff, scheme_coeff):
        self._n = len(mesh.verts)
        self._mesh = mesh
        self._eta = struct_coeff   # np.cos(self._phi)
        self._eps = scheme_coeff
        self.u = self.conf_factor(radius_map)
        self._gamma = radius_map
        self.u = self.conf_factor(radius_map)
        self.u = self.u - sum(self.u)/self._n
        self._theta = dict()
        self._l = dict()
        self.update()

    @classmethod
    def from_riemannian_metric(cls, g, scheme="inversive", _alpha=-1):
        eta = sparse.lil_matrix((g._n, g._n))
        mesh = g._mesh
        gamma = None
        # initial radius
        if scheme=="inversive":
            gamma = np.zeros((g._n,))
            pre_gamma = [[] for _ in g._mesh.verts]
            for face in mesh.faces:
                for i, opp_edge in partition_face(face):
                    j,k = opp_edge
                    pre_gamma[i].append(.5*(g.length((k,i)) + g.length((i,j)) - g.length((j,k))))
            gamma = np.array([min(g_ijk) for g_ijk in pre_gamma])
            # for vert in g._mesh.verts:
            #     gamma[vert] = (1.0/3)*min(g.length(edge) for edge in g._mesh.adjacent_edges(vert))
        elif scheme=="thurston":
            pre_gamma = [[] for _ in g._mesh.verts]
            for face in mesh.faces:
                for i, opp_edge in partition_face(face):
                    j,k = opp_edge
                    pre_gamma[i].append(.5*(g.length((k,i)) + g.length((i,j)) - g.length((j,k))))
            gamma = np.array([(1.0/len(g_ijk))*sum(g_ijk) for g_ijk in pre_gamma])
        elif scheme=="thurston2":
            gamma = np.array(
                [(2.0/3.0)*min(g.length(edge) for edge in g._mesh.adjacent_edges(vert)) for vert in g._mesh.verts])

        if "thurston" in scheme:
            # make circles intersect
            if _alpha<0:
                alpha = 1.0
                for i,j in mesh.edges:
                    alpha = max( 1.1*(g.length((i,j))/(gamma[i]+gamma[j])), alpha)
            else:
                alpha = _alpha
            if alpha>1.0:
                gamma *= alpha
                print("alpha: ",alpha)
                for i,j in mesh.edges:
                    assert gamma[i]+gamma[j] >= g.length((i,j))


        # edge weight
        if scheme=="combinatorial" or isfloat(scheme):
            _eta = float(scheme) if isfloat(scheme) else 1
            print("constant eta: ",_eta)
            gamma = np.ones(g._n)
            for edge in g._mesh.edges:
                i,j = edge
                eta[i,j] = _eta
                eta[j,i] = _eta
        else: # eta from radii
            if gamma is None:
                print("Unknown scheme: ",scheme)
                exit()
            for edge in g._mesh.edges:
                i,j = edge
                struct_c = ((g.length(edge)**2 - gamma[i]**2 - gamma[j]**2) / (2*gamma[i]*gamma[j]))
                #assert(struct_c>=0)
                if "thurston" in scheme:
                    struct_c = np.clip(struct_c,0,1)
                eta[i,j] = struct_c
                eta[j,i] = struct_c

        ret = cls(g._mesh, gamma, eta, IdentityDictMap())

        # This new metric should approximate the old
        #for i,j in mesh.edges:
        #    assert abs(g.length((i,j))-ret._l[(i,j)])<1e-3
        return ret

    def _tau(self, i,j,k):
        return .5*(self._l[j,k]**2 +
                    self._eps[j] * (self._gamma[j]**2) - 
                    self._eps[k] * (self._gamma[k]**2))

    def compute_length(self, edge):
        i,j = list(edge)
        g_i, g_j = self._gamma[[i,j]]
        return np.sqrt(2*g_i*g_j*self._eta[i,j] + self._eps[i] * g_i**2 + self._eps[j] * g_j**2)

    def update(self):
        self._gamma = np.exp(self.u)
        for edge in self._mesh.edges:
            i,j = edge
            l = self.compute_length(edge)
            self._l[i,j] = l
            self._l[j,i] = l
        super(CirclePackingMetric, self).update()

    def hessian(self):
        n = len(self._mesh.verts)
        H = dict() # sparse.dok_matrix((n,n)) ## creating matrix directly costs a lot
        for face in self._mesh.faces:
            i,j,k = face
            A = self.face_area(face)
            L = np.diag((self._l[j,k],self._l[i,k],self._l[i,j]))
            D = self._D(self._tau,i,j,k)
            Theta = self._Theta(face)

            Tijk = -.5/A * (L.dot(Theta).dot(inv(L)).dot(D))
            for a,row in zip((i,j,k), Tijk):
                for b,dtheta in zip((i,j,k), row):
                    if (a,b) in H:
                        H[a,b] += dtheta
                    else:
                        H[a,b] = dtheta
        Hm = sparse.dok_matrix((n,n))
        for (du_i,dtheta_j), val in H.items():
            Hm[du_i, dtheta_j] = val
        return Hm.tocsr()
        # for a,row in zip((i,j,k), Tijk):
        #     for b,dtheta in zip((i,j,k), row):
        #         H[a,b] += dtheta
        # return H

    def ricci_flow(self, target_K=0, dt=0.05, thresh=1e-4, use_hess=False, leave_boundary=False):
        DeltaK = self._K - target_K
        niter = 0
        while (np.abs(DeltaK).sum() > thresh and leave_boundary==False) or (np.abs(DeltaK[self._mesh.free_verts]).sum() > thresh and leave_boundary==True):
            if use_hess:
                if leave_boundary:
                    print("use_hess and leave_boundary together are not implemented!")
                    exit()
                H = self.hessian()
                deltau = sparse.linalg.lsqr(H, DeltaK)[0]
                self.u += dt*deltau
            else:
                if leave_boundary:
                    self.u[self._mesh.free_verts] -= dt*DeltaK[self._mesh.free_verts]
                else:
                    self.u -= dt*DeltaK
            self.u = self.u - sum(self.u)/self._n
            self.update()
            DeltaK = self._K - target_K
            if niter % 100 == 0:
                if leave_boundary:
                    print("niter:", niter, " |DeltaK|_1 %s"%np.abs(DeltaK[self._mesh.free_verts]).sum())
                else:
                    print("niter:", niter, " |DeltaK|_1 %s"%np.abs(DeltaK).sum())
            niter += 1
        return self

    def ricci_flow_op(self, targetK, UfreeV, boundary_u, alpha=1.0, xtol=1e-4, optimizer="lm"):
        target = lambda x: ricciEnergy(x, self._eta, targetK, UfreeV, boundary_u, self._mesh, alpha)
        self.u = least_squares(target, self.u, verbose=2, method=optimizer, xtol=xtol, gtol=xtol).x
        self.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='find metric with a desired curvature by Ricci flow')
    parser.add_argument('input', help='Path to an input ply file')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--gtol', '-gt', type=float, default=1e-6, help='stopping criteria for gradient')
    parser.add_argument('--lambda_bdu', '-lu', type=float, default=1.0, help="weight for boundary constraint")
    parser.add_argument('--method', '-m', default='inversive', help='method for Ricci flow: inversive,thurston,thurston2,combinatorial, or a positive number for a constant eta')
    parser.add_argument('--alpha', '-a', default=-1, type=float, help='multiplication factor of the radius of Thurston circle packing (set to negative for automatic detection)')
    parser.add_argument('--outdir', '-o', default='result',help='Directory to output the result')
    parser.add_argument('--verbose', '-v', action='store_true',help='print debug information')
    parser.add_argument('--optimizer', '-op', default='trf', choices=['sgd','newton','lm','trf'], help='optimiser')
    parser.add_argument('--leave_boundary', '-lb', action='store_true',help='do not touch boundary (keep radii unchanged)')
    parser.add_argument('--target_curvature_interior', '-Ki', default=10, type=float, help='target gaussian curvature value (if >2pi, uniform curvature for internal vertices posed by the Gauss-Bonnet)')
    parser.add_argument('--target_curvature_boundary', '-Kb', default=10, type=float, help='target gaussian curvature value on the boundary vertices (if >2pi, boundary curvature values are fixed to the ones of the initial mesh)')
    parser.add_argument('--target_curvature', '-K', default=None, type=str, help='file containing target gaussian curvature')
    parser.add_argument('--embed', '-e', action='store_true',help='perform embedding as well')
    args = parser.parse_args()

    os.makedirs(args.outdir,exist_ok=True)
    fn, ext = os.path.splitext(args.input)
    fn = os.path.basename(fn).rsplit('_', 1)[0]
    fn = os.path.join(args.outdir,fn)
    if ext == ".obj":
        with open(args.input) as fp:
            v,f = read_obj_file(fp)
    else:
        plydata = PlyData.read(args.input)
        v = np.vstack([plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z']]).astype(np.float64).T
        f = []
        for poly in plydata['face']['vertex_indices']:
            for face in triangulate(poly):
                f.append(face)

    save_ply(v,np.array([list(x) for x in f], dtype=np.int16),fn+".ply")
    mesh = TriangleMesh(v,f)
    g = DiscreteRiemannianMetric(mesh, mesh.lengths)
    
    # set target curvature
    K = np.full(len(v),4*np.pi)
    K[mesh.free_verts] = args.target_curvature_interior
    if args.target_curvature_boundary < 2*np.pi:
        K[mesh.b_verts] = args.target_curvature_boundary
    elif args.target_curvature_boundary < 99:
        K[mesh.b_verts] = g._K[mesh.b_verts] 
    if args.target_curvature:
        tc = np.loadtxt(args.target_curvature,delimiter=",")
        K[tc[:,0].astype(np.uint32)]=tc[:,1]

    if args.target_curvature_interior >= 49: # concentrate at one vertex; demo purpose only
        vid=0
        K[mesh.free_verts] = 0
        K[vid] = (2*mesh.chi()*np.pi - np.sum(K))
        print("flat with curvature concentrating at a vertex: ",K[mesh.free_verts[vid]])
    if args.leave_boundary:
        # save boundary coordinates for later stage
        np.savetxt(fn+"_boundary.csv", np.hstack([np.array(mesh.b_verts)[:,np.newaxis],v[mesh.b_verts]]),delimiter=",", fmt='%i,%f,%f,%f')

    ## for unssigned vertices, set uniform values from Gauss-Bonnet
    KfreeV = K>2*np.pi
    if sum(KfreeV)>0:
        K[KfreeV] = (2*mesh.chi()*np.pi - np.sum(K[~KfreeV]))/sum(KfreeV)

    # vertices to determine u
    if args.leave_boundary:
        UfreeV = mesh.free_verts
    else:
        UfreeV = mesh.verts

    np.savetxt(fn+"_innerVertexID.txt",mesh.free_verts,fmt='%i')
    np.savetxt(fn+"_targetCurvature.txt", K)
    print("#V: %s, #E: %s, #F: %s, Min vertex valence: %s" % (len(mesh.verts),len(mesh.edges),len(mesh.faces), mesh.min_valence()))
    print("Mesh chi: %s, global chi: %s, boundary curvature: %s, target total curvature: %s pi" % (mesh.chi(),g.gb_chi(),K[mesh.b_verts].sum(), K.sum()/np.pi))

    # ricci flow
    cp = CirclePackingMetric.from_riemannian_metric(g, scheme=args.method, _alpha=args.alpha)

    init_u = cp.u.copy()
    init_E = ricciEnergy(cp.u, cp._eta, K, UfreeV, init_u[mesh.b_verts], mesh, 1)

    start = time.time()
    if args.optimizer == "sgd":
        cp.ricci_flow(target_K=K, dt=args.learning_rate, thresh=args.gtol, use_hess=False, leave_boundary=args.leave_boundary)
    elif args.optimizer == "newton":
        cp.ricci_flow(target_K=K, dt=args.learning_rate, thresh=args.gtol, use_hess=True, leave_boundary=args.leave_boundary)
    else:
        cp.ricci_flow_op(K, UfreeV, init_u[mesh.b_verts], alpha=args.lambda_bdu, xtol=args.gtol, optimizer=args.optimizer)
    print ("{} sec".format(time.time() - start))

    final_E = ricciEnergy(cp.u, cp._eta, K, UfreeV, init_u[mesh.b_verts], mesh, 1)
    print("Ricci Energy initial: {}, final: {}".format(np.abs(init_E[:len(UfreeV)]**2).sum(),np.abs(final_E[:len(UfreeV)]**2).sum() ))

    edgedata = np.array( [[i,j,l] for (i,j), l in cp._l.items()] )
    np.savetxt(fn+"_edge.csv", edgedata,delimiter=",",fmt='%i,%i,%f')
    sns.violinplot(y=cp._K[UfreeV], cut=0)
    print("total curvature error: {}, boundary error: {}".format(np.abs(cp._K[UfreeV]-K[UfreeV]).sum(),np.abs(final_E[len(UfreeV):]).sum() ))
    plt.savefig(fn+"_curvature_ricci.png")
    plt.close()

    if args.embed:
        dn = os.path.dirname(__file__)
        cmd = "python {} {}".format(os.path.join(dn,"metric_embed.py"), fn+".ply")
        print("\n",cmd)
        subprocess.call(cmd, shell=True)
        cmd = "python {} {}".format(os.path.join(dn,"evaluation.py"),fn+"_final.ply")
        print("\n",cmd)
        subprocess.call(cmd, shell=True)
