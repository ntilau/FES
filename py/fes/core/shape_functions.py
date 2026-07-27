"""Scalar and vector shape functions on reference intervals and triangles.

This is a direct port of CalcShapeFunctions.m / CalcShapeFunctionsSymbolic.m.
Returns numpy-friendly lambda functions that evaluate the basis and their
derivatives at a given reference coordinate.
"""

import numpy as np


def _shape_1d(order):
    """Return (shape_func, deriv_func) for 1D reference interval [0,1]."""
    if order == 1:
        def sf(x):
            return np.array([1 - x, x])
        def df(x):
            return np.array([-1.0, 1.0])
        return sf, df
    elif order == 2:
        def sf(x):
            return np.array([
                (2*x - 1)*(x - 1),
                x*(2*x - 1),
                -4*x*(x - 1)
            ])
        def df(x):
            return np.array([
                4*x - 3,
                4*x - 1,
                4 - 8*x
            ])
        return sf, df
    elif order == 3:
        def sf(x):
            return np.array([
                -((3*x - 1)*(3*x - 2)*(x - 1))/2,
                (x*(3*x - 1)*(3*x - 2))/2,
                (9*x*(3*x - 2)*(x - 1))/2,
                -(9*x*(3*x - 1)*(x - 1))/2
            ])
        def df(x):
            return np.array([
                18*x - (27*x**2)/2 - 11/2,
                (27*x**2)/2 - 9*x + 1,
                (81*x**2)/2 - 45*x + 9,
                36*x - (81*x**2)/2 - 9/2
            ])
        return sf, df
    elif order == 4:
        def sf(x):
            return np.array([
                ((2*x - 1)*(4*x - 1)*(4*x - 3)*(x - 1))/3,
                (x*(2*x - 1)*(4*x - 1)*(4*x - 3))/3,
                -(16*x*(2*x - 1)*(4*x - 3)*(x - 1))/3,
                4*x*(4*x - 1)*(4*x - 3)*(x - 1),
                -(16*x*(2*x - 1)*(4*x - 1)*(x - 1))/3
            ])
        def df(x):
            return np.array([
                ((8*x - 5)*(16*x**2 - 20*x + 5))/3,
                ((8*x - 3)*(16*x**2 - 12*x + 1))/3,
                -(512*x**3)/3 + 288*x**2 - (416*x)/3 + 16,
                4*(2*x - 1)*(32*x**2 - 32*x + 3),
                -(512*x**3)/3 + 224*x**2 - (224*x)/3 + 16/3
            ])
        return sf, df
    else:
        raise ValueError(f"Invalid 1D order: {order}")


def _shape_2d(order):
    """Return (shape_func, deriv_x_func, deriv_y_func) for reference triangle.

    Reference triangle: vertices at (0,0), (1,0), (0,1) in (x,y) space.
    """
    if order == 1:
        def sf(x, y):
            return np.array([1 - y - x, x, y])
        def dx(x, y):
            return np.array([-1.0, 1.0, 0.0])
        def dy(x, y):
            return np.array([-1.0, 0.0, 1.0])
        return sf, dx, dy
    elif order == 2:
        def sf(x, y):
            return np.array([
                (2*x + 2*y - 1)*(x + y - 1),
                x*(2*x - 1),
                y*(2*y - 1),
                4*x*y,
                -4*y*(x + y - 1),
                -4*x*(x + y - 1)
            ])
        def dx(x, y):
            return np.array([
                4*x + 4*y - 3,
                4*x - 1,
                0.0,
                4*y,
                -4*y,
                4 - 4*y - 8*x
            ])
        def dy(x, y):
            return np.array([
                4*x + 4*y - 3,
                0.0,
                4*y - 1,
                4*x,
                4 - 8*y - 4*x,
                -4*x
            ])
        return sf, dx, dy
    elif order == 3:
        def sf(x, y):
            return np.array([
                -((3*x + 3*y - 1)*(3*x + 3*y - 2)*(x + y - 1))/2,
                (x*(3*x - 1)*(3*x - 2))/2,
                (y*(3*y - 1)*(3*y - 2))/2,
                (9*x*y*(3*x - 1))/2,
                (9*y*(3*x + 3*y - 2)*(x + y - 1))/2,
                (9*x*(3*x + 3*y - 2)*(x + y - 1))/2,
                (9*x*y*(3*y - 1))/2,
                -(9*y*(3*y - 1)*(x + y - 1))/2,
                -(9*x*(3*x - 1)*(x + y - 1))/2,
                -27*x*y*(x + y - 1)
            ])
        def dx(x, y):
            return np.array([
                18*x + 18*y - 27*x*y - (27*x**2)/2 - (27*y**2)/2 - 11/2,
                (27*x**2)/2 - 9*x + 1,
                0.0,
                (9*y*(6*x - 1))/2,
                (9*y*(6*x + 6*y - 5))/2,
                (81*x**2)/2 + 54*x*y - 45*x + (27*y**2)/2 - (45*y)/2 + 9,
                (9*y*(3*y - 1))/2,
                -(9*y*(3*y - 1))/2,
                36*x + (9*y)/2 - 27*x*y - (81*x**2)/2 - 9/2,
                -27*y*(2*x + y - 1)
            ])
        def dy(x, y):
            return np.array([
                18*x + 18*y - 27*x*y - (27*x**2)/2 - (27*y**2)/2 - 11/2,
                0.0,
                (27*y**2)/2 - 9*y + 1,
                (9*x*(3*x - 1))/2,
                (27*x**2)/2 + 54*x*y - (45*x)/2 + (81*y**2)/2 - 45*y + 9,
                (9*x*(6*x + 6*y - 5))/2,
                (9*x*(6*y - 1))/2,
                (9*x)/2 + 36*y - 27*x*y - (81*y**2)/2 - 9/2,
                -(9*x*(3*x - 1))/2,
                -27*x*(x + 2*y - 1)
            ])
        return sf, dx, dy
    elif order == 4:
        def sf(x, y):
            return np.array([
                ((2*x + 2*y - 1)*(4*x + 4*y - 1)*(4*x + 4*y - 3)*(x + y - 1))/3,
                (x*(2*x - 1)*(4*x - 1)*(4*x - 3))/3,
                (y*(2*y - 1)*(4*y - 1)*(4*y - 3))/3,
                (16*x*y*(2*x - 1)*(4*x - 1))/3,
                -(16*y*(2*x + 2*y - 1)*(4*x + 4*y - 3)*(x + y - 1))/3,
                -(16*x*(2*x + 2*y - 1)*(4*x + 4*y - 3)*(x + y - 1))/3,
                4*x*y*(4*x - 1)*(4*y - 1),
                4*y*(4*y - 1)*(4*x + 4*y - 3)*(x + y - 1),
                4*x*(4*x - 1)*(4*x + 4*y - 3)*(x + y - 1),
                32*x*y*(4*x + 4*y - 3)*(x + y - 1),
                (16*x*y*(2*y - 1)*(4*y - 1))/3,
                -(16*y*(2*y - 1)*(4*y - 1)*(x + y - 1))/3,
                -(16*x*(2*x - 1)*(4*x - 1)*(x + y - 1))/3,
                -32*x*y*(4*x - 1)*(x + y - 1),
                -32*x*y*(4*y - 1)*(x + y - 1)
            ])
        def dx(x, y):
            return np.array([
                ((8*x + 8*y - 5)*(16*x**2 + 32*x*y - 20*x + 16*y**2 - 20*y + 5))/3,
                ((8*x - 3)*(16*x**2 - 12*x + 1))/3,
                0.0,
                (16*y*(24*x**2 - 12*x + 1))/3,
                -(16*y*(24*x**2 + 48*x*y - 36*x + 24*y**2 - 36*y + 13))/3,
                -512*x**3/3 - 384*x**2*y + 288*x**2 - 256*x*y**2 + 384*x*y - 416*x/3 - 128*y**3/3 + 96*y**2 - 208*y/3 + 16,
                4*y*(8*x - 1)*(4*y - 1),
                4*y*(4*y - 1)*(8*x + 8*y - 7),
                4*(2*x + y - 1)*(32*x*y - 4*y - 32*x + 32*x**2 + 3),
                32*y*(12*x**2 + 16*x*y - 14*x + 4*y**2 - 7*y + 3),
                (16*y*(2*y - 1)*(4*y - 1))/3,
                -(16*y*(2*y - 1)*(4*y - 1))/3,
                64*x*y - 16*y/3 - 224*x/3 - 128*x**2*y + 224*x**2 - 512*x**3/3 + 16/3,
                -32*y*(8*x*y - y - 10*x + 12*x**2 + 1),
                -32*y*(4*y - 1)*(2*x + y - 1)
            ])
        def dy(x, y):
            return np.array([
                ((8*x + 8*y - 5)*(16*x**2 + 32*x*y - 20*x + 16*y**2 - 20*y + 5))/3,
                0.0,
                ((8*y - 3)*(16*y**2 - 12*y + 1))/3,
                (16*x*(2*x - 1)*(4*x - 1))/3,
                -128*x**3/3 - 256*x**2*y + 96*x**2 - 384*x*y**2 + 384*x*y - 208*x/3 - 512*y**3/3 + 288*y**2 - 416*y/3 + 16,
                -(16*x*(24*x**2 + 48*x*y - 36*x + 24*y**2 - 36*y + 13))/3,
                4*x*(4*x - 1)*(8*y - 1),
                4*(x + 2*y - 1)*(32*x*y - 32*y - 4*x + 32*y**2 + 3),
                4*x*(4*x - 1)*(8*x + 8*y - 7),
                32*x*(4*x**2 + 16*x*y - 7*x + 12*y**2 - 14*y + 3),
                (16*x*(24*y**2 - 12*y + 1))/3,
                64*x*y - 224*y/3 - 16*x/3 - 128*x*y**2 + 224*y**2 - 512*y**3/3 + 16/3,
                -(16*x*(2*x - 1)*(4*x - 1))/3,
                -32*x*(4*x - 1)*(x + 2*y - 1),
                -32*x*(8*x*y - 10*y - x + 12*y**2 + 1)
            ])
        return sf, dx, dy
    else:
        raise ValueError(f"Invalid 2D order: {order}")


def get_shape_functions(dim, order):
    """Get shape functions and their derivatives.

    Parameters
    ----------
    dim : int
        Spatial dimension (1 or 2).
    order : int
        Polynomial order (1-4).

    Returns
    -------
    tuple
        For dim=1: (shape_func, deriv_func)
        For dim=2: (shape_func, deriv_x_func, deriv_y_func)
    """
    if dim == 1:
        return _shape_1d(order)
    elif dim == 2:
        return _shape_2d(order)
    else:
        raise ValueError(f"Invalid dimension: {dim}")


def calc_curl_shape_functions(quad_pts, order, inv_jt=None):
    """Compute H(curl)-conforming vector basis on a triangle.

    Converts scalar shape functions of order 1 into vector-valued
    edge/face basis functions. If inv_jt is provided, the result
    is transformed to physical coordinates.

    Parameters
    ----------
    quad_pts : ndarray, shape (nq, 2)
        Quadrature points in reference coordinates (xi, eta).
    order : int
        Polynomial order.
    inv_jt : ndarray, shape (2, 2)
        Inverse transpose of the Jacobian (for physical coords).

    Returns
    -------
    Nv : ndarray, shape (nq, numv, 2)
        Vector basis functions at each quadrature point.
    dNv : ndarray, shape (nq, numv)
        Divergence of vector basis functions.
    """
    import numpy as np

    nq = quad_pts.shape[0]

    if inv_jt is None:
        inv_jt = np.eye(2)

    # Get linear shape functions for vector construction
    sf1, dx1, dy1 = get_shape_functions(2, 1)
    numv = calc_order_mat_size(order)[1]

    Nv = np.zeros((nq, numv, 2))
    dNv = np.zeros((nq, numv))

    for iq in range(nq):
        x, y = quad_pts[iq, 0], quad_pts[iq, 1]
        xs = sf1(x, y)      # shape values at quad pt
        dxs = dx1(x, y)     # shape derivatives wrt ref coords
        dys = dy1(x, y)

        # Transform to physical derivatives
        dxy = inv_jt @ np.array([dxs, dys])  # (2, 3)
        dxp = dxy[0, :]
        dyp = dxy[1, :]

        if order == 1:
            Nv_i = np.zeros((3, 2))
            Nv_i[0, :] = xs[1] * dxy[:, 2] - xs[2] * dxy[:, 1]
            Nv_i[1, :] = xs[0] * dxy[:, 2] - xs[2] * dxy[:, 0]
            Nv_i[2, :] = xs[0] * dxy[:, 1] - xs[1] * dxy[:, 0]
            Nv[iq] = Nv_i
            dNv[iq, :] = np.array([2.0, -2.0, 2.0])
        elif order == 2:
            Nv_i = np.zeros((8, 2))
            # edge-based functions (order 1 type)
            Nv_i[0, :] = xs[1] * dxy[:, 2] - xs[2] * dxy[:, 1]
            Nv_i[1, :] = xs[0] * dxy[:, 2] - xs[2] * dxy[:, 0]
            Nv_i[2, :] = xs[0] * dxy[:, 1] - xs[1] * dxy[:, 0]
            # quadratic edge bubbles
            Nv_i[3, :] = 4 * (xs[1] * dxy[:, 2] + xs[2] * dxy[:, 1])
            Nv_i[4, :] = 4 * (xs[0] * dxy[:, 2] + xs[2] * dxy[:, 0])
            Nv_i[5, :] = 4 * (xs[0] * dxy[:, 1] + xs[1] * dxy[:, 0])
            # quadratic face bubbles
            Nv_i[6, :] = xs[0] * xs[1] * dxy[:, 2] - xs[0] * xs[2] * dxy[:, 1]
            Nv_i[7, :] = xs[0] * xs[1] * dxy[:, 2] - xs[1] * xs[2] * dxy[:, 0]
            Nv[iq] = Nv_i
            dNv[iq, :] = np.array([
                2.0, -2.0, 2.0,
                0.0, 0.0, 0.0,
                2*xs[0] - xs[1] - xs[2],
                xs[0] + xs[2] - 2*xs[1]
            ])
        elif order == 3:
            Nv_i = np.zeros((15, 2))
            # edge functions (order 1 type)
            Nv_i[0, :] = xs[1] * dxy[:, 2] - xs[2] * dxy[:, 1]
            Nv_i[1, :] = xs[0] * dxy[:, 2] - xs[2] * dxy[:, 0]
            Nv_i[2, :] = xs[0] * dxy[:, 1] - xs[1] * dxy[:, 0]
            # quadratic edge bubbles
            Nv_i[3, :] = 4 * (xs[1] * dxy[:, 2] + xs[2] * dxy[:, 1])
            Nv_i[4, :] = 4 * (xs[0] * dxy[:, 2] + xs[2] * dxy[:, 0])
            Nv_i[5, :] = 4 * (xs[0] * dxy[:, 1] + xs[1] * dxy[:, 0])
            # quadratic face bubbles
            Nv_i[6, :] = xs[0] * xs[1] * dxy[:, 2] - xs[0] * xs[2] * dxy[:, 1]
            Nv_i[7, :] = xs[0] * xs[1] * dxy[:, 2] - xs[1] * xs[2] * dxy[:, 0]
            # cubic edge functions
            Nv_i[8, :] = xs[1]*(xs[1]-2*xs[2])*dxy[:, 2] + xs[2]*(-xs[2]+2*xs[1])*dxy[:, 1]
            Nv_i[9, :] = xs[0]*(xs[0]-2*xs[2])*dxy[:, 2] + xs[2]*(-xs[2]+2*xs[0])*dxy[:, 0]
            Nv_i[10, :] = xs[0]*(xs[0]-2*xs[1])*dxy[:, 1] + xs[1]*(-xs[1]+2*xs[0])*dxy[:, 0]
            Nv_i[11, :] = xs[1]*xs[0]*dxy[:, 2] + xs[0]*xs[2]*dxy[:, 1] + xs[1]*xs[2]*dxy[:, 0]
            # cubic face bubbles
            Nv_i[12, :] = -xs[1]*xs[0]*(xs[1]-2*xs[2])*dxy[:, 2] - xs[0]*xs[2]*(-xs[2]+2*xs[1])*dxy[:, 1] + 3*xs[1]*xs[2]*(xs[1]-xs[2])*dxy[:, 0]
            Nv_i[13, :] = -xs[0]*xs[1]*(xs[0]-2*xs[2])*dxy[:, 2] + 3*xs[0]*xs[2]*(xs[0]-xs[2])*dxy[:, 1] - xs[1]*xs[2]*(-xs[2]+2*xs[0])*dxy[:, 0]
            Nv_i[14, :] = 3*xs[0]*xs[1]*(xs[0]-xs[1])*dxy[:, 2] - xs[0]*xs[2]*(xs[0]-2*xs[1])*dxy[:, 1] - xs[1]*xs[2]*(-xs[1]+2*xs[0])*dxy[:, 0]
            Nv[iq] = Nv_i
            det = dxp[1]*dyp[2] - dyp[1]*dxp[2]
            dNv[iq, :] = np.array([
                2.0, -2.0, 2.0,
                0.0, 0.0, 0.0,
                2*xs[0] - xs[1] - xs[2],
                xs[0] + xs[2] - 2*xs[1],
                0.0, 0.0, 0.0, 0.0,
                -16*xs[1]*xs[2] + 4*xs[2]**2 + 4*xs[1]**2,
                16*xs[0]*xs[2] - 4*xs[2]**2 - 4*xs[0]**2,
                -16*xs[1]*xs[0] + 4*xs[1]**2 + 4*xs[0]**2
            ]) * (dxp[1]*dyp[2] - dyp[1]*dxp[2])
        else:
            raise ValueError(f"Vector shape functions not implemented for order {order}")

    return Nv, dNv


def calc_order_mat_size(order):
    """Return the number of scalar and vector DOFs per element for given order.

    Port of CalcOrderMatSize.m.

    Parameters
    ----------
    order : int
        Polynomial order.

    Returns
    -------
    nums : int
        Number of scalar DOFs per element.
    numv : int
        Number of vector DOFs per element.
    """
    nums = (order + 1) * (order + 2) // 2
    numv = order * (order + 2)
    return nums, numv
