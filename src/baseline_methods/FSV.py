from sklearn.feature_selection import mutual_info_classif
import numpy as np
from scipy.stats import spearmanr


eps = np.finfo(float).eps


def slinearsolve(c, A, b, C):
    flag = 0
    m, n = A.shape
    mu = 1
    maxiter = 50
    itref = 1
    cont = True
    clean_eps = 1e-8
    trdoff = 0
    alpha = np.zeros((n, 1))

    x = 100 * np.ones((n, 1))
    z = 100 * np.ones((n, 1))
    y = 100 * np.ones((m, 1))
    if C < np.Inf:
        s = 100 * np.ones((n, 1))
        w = 100 * np.ones((n, 1))
    else:
        s = []
        w = []

    dinf = 1e-14
    smallz = 1e-14
    smallt = 23e-17
    opttol = 1e-6

    if C < np.Inf:
        u = np.ones((n, 1))
        b = b / C

    objQ = .5 * trdoff * x.T @ x

    c -= trdoff * alpha

    pobjo = np.abs(c.T @ x + objQ) + 1
    if C < np.Inf:
        dobjo = np.abs(-objQ + b.T @ y - u.T @ w) + 1
    else:
        dobjo = np.abs(b.T @ y - objQ) + 1

    while cont and itref <= maxiter:
        objQ = .5 * trdoff * x.T @ x
        pobj = c.T @ x + objQ
        if C < np.Inf:
            dobj = b.T @ y - u.T @ w - objQ
        else:
            dobj = b.T @ y - objQ

        dlgap = pobj - dobj
        dp = np.abs(pobj) / (np.abs(pobjo) + 1.)
        dd = np.abs(dobj) / (np.abs(dobjo) + 1.)
        dobjo = dobj
        pobjo = pobj

        if dp > 1e6:
            flag = -1
            print('Solution not bounded in the primal. Exit.')
            return x,y,flag,z,s,w
        if dd > 1e6:
            flag = -1
            print('Solution not bounded in the dual. Exit.')
            return x,y,flag,z,s,w

        oldgap = dlgap
        dp = np.abs(dobj) + 1.
        if np.abs(dlgap) / dp <= opttol and itref > 1:
            cont = False
            break

        dp = dp + np.abs(pobj)
        T = np.abs(dlgap) / dp

        if itref <= 3:
            ax = 2e-3
            az = 1e-3
        elif T >= 0.8:
            ax = 2e-4
            az = 1e-4
        elif T >= 0.1:
            ax = 2e-5
            az = 1e-5
        elif T >= 0.01:
            ax = 2e-6
            az = 1e-6
        elif T >= 0.001:
            ax = 2e-7
            az = 1e-7
        elif T >= 0.0001:
            ax = 2e-8
            az = 1e-7
        elif T >= 0.00001:
            ax = 2e-9
            az = 1e-9
        else:
            ax = T * 1e-5
            az = ax

        x = x + ax
        z = z + az
        if C < np.Inf:
            s = s + ax
            w = w + az

        xi_b = -A @ x + b
        xi_c = c - A.T @ y - z + trdoff*x
        xi_z = - x * z
        if C < np.Inf:
            xi_c = xi_c + w
            xi_u = u - x - s
            xi_w = - s * w

        if C < np.Inf:
            dp = x
            if np.max(np.abs(dp)) <= smallz:
                print('Conditioning problem to invert theta. Abort.')
                return x,y,flag,z,s,w
            dpp = s
            if np.max(np.abs(dpp)) <= smallz:
                print('Conditioning problem to invert theta. Abort.')
                return x,y,flag,z,s,w
            theta = z / dp + w / dpp

        if C == np.Inf:
            dp = x
            if np.max(np.abs(dp)) <= smallz:
                print('Conditioning problem to invert theta. Abort')
                return x,y,flag,z,s,w
            theta = z / dp

        tmp = np.ones_like(theta)
        theta = tmp / (trdoff * tmp + theta)

        neglect = np.where(theta < smallt)[0]
        if len(neglect) > 0:
            theta[neglect] = 0.

        neglect = np.where(theta >= 1e8)[0]
        if len(neglect) > 0:
            theta[neglect] = 1e4 * np.sqrt(theta[neglect])

        if C < np.Inf:
            f = xi_c - xi_z / x + (xi_w - xi_u * w) / s
            h = xi_b
        else:
            f = xi_c - xi_z / x
            h = xi_b

        AA = np.zeros_like(A)
        for i in range(len(A)):
            AA[i] = A[i] * theta.T
        to_inv = AA @ f + h
        AAA = AA @ A.T
        dy = np.linalg.inv(to_inv) @ AAA  # AAA\to_inv;
        dx = AA @ dy - f * theta
        dz = (xi_z - z * dx) / x
        if C < np.Inf:
            ds = xi_u - dx
            dw = (xi_w - w * ds) / s

        indz = np.where(dz < 0)[0]
        indx = np.where(dx < 0)[0]
        inds = []
        mins = 1
        indw = []
        minw = 1
        if C < np.Inf:
            inds = np.where(ds < 0)[0]
            indw = np.where(dw < 0)[0]
            if len(inds) > 0:
                mins = np.min(-(s[inds] - dinf) / ds[inds])
            else:
                mins = 1
            if len(indw) > 0:
                minw = np.min(-(w[indw] - dinf) / dw[indw])
            else:
                minw = 1
        if len(indx) > 0.:
            minx = np.min(-(x[indx] - dinf) / dx[indx])
        else:
            minx = 1

        apk = np.min([minx, mins, 1])
        if len(indz) > 0:
            minz = np.min(-(z[indz] - dinf) / dz[indz])
        else:
            minz = 1

        adk = np.min([minw, minz, 1])

        ax = np.sum(x * z)
        nas = np.sum((x + apk * dx) * (z + adk * dz))
        az = np.sum(dx ** 2 + dz ** 2)
        if C < np.Inf:
            ax = ax + np.sum(s * w)
            nas = nas + np.sum((s + apk * ds) * (w + adk * dw))
            az = az + np.sum(ds ** 2 + dw ** 2)

        if nas <= opttol:
            cont = 0
            x = x + apk * dx
            y = y + adk * dy
            z = z + adk * dz
            if C < np.Inf:
                s = s + apk * ds
                w = w + adk * dw
            break

        mu = (nas / ax) ** 2 * nas / n

        xi_z = - x * z + mu * np.ones_like(x) - dx * dz
        f = xi_c - xi_z / x
        if C < np.Inf:
            xi_w = - s * w + mu * np.ones_like(s) - ds * dw
            f = f + xi_w / s - (w * xi_u) / s
        h = xi_b

        AA = np.zeros_like(A)
        for i in range(len(A)):
            AA[i] = A[i] * theta.T
        to_inv = AA @ f + h
        AAA = AA @ A.T
        dy = np.linalg.inv(to_inv) @ AAA  # AAA\to_inv;
        dx = AA.T @ dy - f * theta
        dz = (xi_z - z * dx) / x
        if C < np.Inf:
            ds = xi_u - dx
            dw = (xi_w - w * ds) / s

        indz = np.where(dz < 0)[0]
        indx = np.where(dx < 0)[0]
        if C < np.Inf:
            inds = np.where(ds < 0)[0]
            indw = np.where(dw < 0)[0]
            if len(inds) > 0:
                mins = np.min(-s[inds] / ds[inds])
            else:
                mins = 1
            if len(indw) > 0:
                minw = np.min(-w[indw] / dw[indw])
            else:
                minw = 1
        if len(indx) > 0:
            minx = np.min(-x[indx] / dx[indx])
        else:
            minx = 1
        alpha_p = np.min([minx, mins, 1])
        if len(indz) > 0:
            minz = np.min(-z[indz] / dz[indz])
        else:
            minz = 1
        alpha_d = np.min([minw, minz, 1])
        spd = min(alpha_p, alpha_d)
        fp = 0.95 * spd
        fd = 0.95 * spd
        x = x + fp * dx
        y = y + fd * dy
        z = z + fd * dz
        if C < np.Inf:
            w = w + fd * dw
            s = s + fp * ds
        itref += 1

    if not cont:
        flag = 1

    tmp = np.where((x < clean_eps / n) | (x < 100 * eps))[0]
    if len(tmp) > 0:
        x[tmp] = 0.

    if C < np.Inf:
        x = C * x

    return x, y, flag, z, s, w


def get_FSV_score(X, y, numF):
    loop = 0
    finished = False
    alpha = 5  # Default
    m, n = X.shape
    v = np.zeros((n, 1))
    w = np.zeros(n)
    y_column = y.reshape((-1, 1))

    while not finished:
        loop += 1
        scale = alpha * np.exp(-alpha * v)

        A = np.concatenate([y_column * X, -y_column * X, y_column, -y_column, -np.eye(m)], axis=1)
        Obj = np.concatenate([scale.T, scale.T, np.zeros((1, m)), np.zeros((1, m)), np.zeros((1, m))], axis=1)
        b = np.ones((m, 1))
        x, _, _, _, _, _ = slinearsolve(Obj.T, A, b, np.Inf)
        w = x[:n] - x[n:2*n]
        b0 = x[2 * n] - x[2 * n + 1]
        vnew = np.abs(w)

        if np.linalg.norm(vnew - v, 1) < 1e-5 * np.linalg.norm(v, 1):
            finished = True
        else:
            v = vnew

        if loop > 2:
            finished = True
        nfeat = len(np.where(vnew > 100 * eps)[0])

        print('Iter ', loop, ' - feat ', nfeat)

        if nfeat < numF:
            finished = True

    return w


class FSV:

    def __init__(self, n_features_to_select=10):
        self.n_features_to_select = n_features_to_select
        self.score = None
        self.ranking = None

    def fit(self, X, y):
        self.score = get_FSV_score(X, y, self.n_features_to_select)
        self.ranking = np.argsort(-self.score, 0)

    def transform(self, X):
        new_X = X[:, self.ranking[:self.n_features_to_select]]
        return new_X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)