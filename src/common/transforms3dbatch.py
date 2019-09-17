import numpy as np

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# For testing whether a number is close to zero
_EPS4 = np.finfo(float).eps * 4.0

_FLOAT_EPS = np.finfo(np.float).eps


def quat2matbatch(q):
    ''' Calculate rotation matrix corresponding to quaternion

    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows quaternions that
    have not been normalized.

    References
    ----------
    Algorithm from http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    assert q.shape[-1] == 4, 'last dimension must be 4'
    original_shape = list(q.shape)
    original_shape = original_shape[:-1] + [3,3]
    q = q.reshape(-1, 4)
    
    eyes = np.stack([np.eye(3) for _ in range(q.shape[0])], axis=0)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3] 
    Nq = w*w + x*x + y*y + z*z
    mask = Nq < _FLOAT_EPS
    mask = mask.astype(np.int)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s

    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    mat = np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])
    mat = mat.transpose(2, 0, 1)
    mat = (1 - mask[:, np.newaxis, np.newaxis]) * mat + mask[:, np.newaxis, np.newaxis] * eyes
    return mat.reshape(*original_shape)

def mat2eulerbatch(mat, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    Note that many Euler angle triplets can describe one matrix.

    Parameters
    ----------
    mat : array-like shape (3, 3) or (4, 4)
        Rotation matrix or affine.
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).

    Examples
    --------
    >>> R0 = euler2mat(1, 2, 3, 'syxz')
    >>> al, be, ga = mat2euler(R0, 'syxz')
    >>> R1 = euler2mat(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True
    """
    assert mat.shape[-1] == 3 & mat.shape[-2] == 3, 'last two dimensions must be 3'
    original_shape = list(mat.shape)
    original_shape = original_shape[:-1]
    mat = mat.reshape(-1, 3, 3)
    
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(mat, dtype=np.float64, copy=False)[:, :3, :3]
    if repetition:
        sy = np.sqrt(M[:, i, j]*M[:, i, j] + M[:, i, k]*M[:, i, k])
        mask = (sy > _EPS4).astype(np.int)
        ax1 = np.arctan2( M[:, i, j],  M[:, i, k])
        ay1 = np.arctan2( sy,       M[:, i, i])
        az1 = np.arctan2( M[:, j, i], -M[:, k, i])
        
        ax2 = np.arctan2(-M[:, j, k],  M[:, j, j])
        ay2 = np.arctan2( sy,       M[:, i, i])
        az2 = np.zeros((M.shape[0],))
        ax = ax1*mask + ax2*(1-mask)
        ay = ay1*mask + ay2*(1-mask)
        az = az1*mask + az2*(1-mask)
    else:
        cy = np.sqrt(M[:, i, i]*M[:, i, i] + M[:, j, i]*M[:, j, i])
        mask = (cy > _EPS4).astype(np.int)
        ax1 = np.arctan2( M[:, k, j],  M[:, k, k])
        ay1 = np.arctan2(-M[:, k, i],  cy)
        az1 = np.arctan2( M[:, j, i],  M[:, i, i])
        
        ax2 = np.arctan2(-M[:, j, k],  M[:, j, j])
        ay2 = np.arctan2(-M[:, k, i],  cy)
        az2 = np.zeros((M.shape[0],))
        ax = ax1*mask + ax2*(1-mask)
        ay = ay1*mask + ay2*(1-mask)
        az = az1*mask + az2*(1-mask)
    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
        
    return np.stack([ax, ay, az], axis=1).reshape(*original_shape)


def quat2eulerbatch(q, axes='sxyz'):
  return mat2eulerbatch(quat2matbatch(q), axes) 

def euler2matbatch(e, axes='sxyz'):
  return quat2matbatch(euler2quatbatch(e, axes=axes))

def mat2quatbatch(m):
  return euler2quatbatch(mat2eulerbatch(m, axes='sxyz'), axes='sxyz')

def euler2quatbatch(e, axes='sxyz'):
    """Return `quaternion` from Euler angles and axis sequence `axes`

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format

    Examples
    --------
    >>> q = euler2quat(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
    True
    """
    assert e.shape[-1] == 3, 'last dimension must be 3'
    original_shape = list(e.shape)
    original_shape = original_shape[:-1] + [4]
    e = e.reshape(-1, 3)
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i+parity-1] + 1
    k = _NEXT_AXIS[i-parity] + 1

    ai, aj, ak = e[:, 0], e[:, 1], e[:, 2]
    
    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj
    
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((e.shape[0], 4, ))
    if repetition:
        q[:, 0] = cj*(cc - ss)
        q[:, i] = cj*(cs + sc)
        q[:, j] = sj*(cc + ss)
        q[:, k] = sj*(cs - sc)
    else:
        q[:, 0] = cj*cc + sj*ss
        q[:, i] = cj*sc - sj*cs
        q[:, j] = cj*ss + sj*cc
        q[:, k] = cj*cs - sj*sc
    if parity:
        q[:, j] *= -1.0

    return q.reshape(*original_shape)
