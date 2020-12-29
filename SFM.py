import numpy as np
from math import sqrt


def calc_TFL_dist(prev, curr, focal, pp, EM):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev, curr, focal, pp, EM)

    if abs(tZ) < 10e-6:
        print('tz = ', tZ)

    elif norm_prev_pts.size == 0:
        print('no prev points')

    elif norm_curr_pts.size == 0:
        print('no curr points')

    else:
        curr.traffic_lights_3d_location, curr.valid = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ)

    return curr


def prepare_3D_data(prev, curr, focal, pp, EM):
    # print("3d",prev.traffic_light, focal, pp[0] )

    norm_prev_pts = normalize(prev.tfl, focal, pp)
    norm_curr_pts = normalize(curr.tfl, focal, pp)
    R, foe, tZ = decompose(np.array(EM))

    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    validVec = []

    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)

        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))

    return np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    return np.array([np.array([tfl[0] - pp[0], tfl[1] - pp[1], focal]) / focal for tfl in pts])


def unnormalize(pts, focal, pp):
    return np.array([np.array([tfl[0] * focal + pp[0], tfl[1] * focal + pp[1], focal]) for tfl in pts])


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion

    # R = EM[:3, :3]
    # t = EM[:3, 3]
    # foe = [t[0] / t[2], t[1] / t[2]]
    # tZ = t[2]
    #
    # return R, foe, tZ
    t = EM[:3, 3]
    return EM[:3, :3], [t[0] / t[2], t[1] / t[2]], t[2]


def rotate(pts, R):
    return np.array([R @ p for p in pts])


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index

    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = (p[1] * foe[0]) - (foe[1] * p[0]) / (foe[0] - p[0])
    d = [abs((m * tfl[0] + n - tfl[1]) / sqrt(pow(m, 2) + 1)) for tfl in norm_pts_rot]
    min_dist = np.argmin(d)

    return min_dist, norm_pts_rot[min_dist]


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z

    zX = (tZ * (foe[0] - p_rot[0])) / (p_curr[0] - p_rot[0])
    zY = (tZ * (foe[1] - p_rot[1])) / (p_curr[1] - p_rot[1])

    # dx_plus_dy = (p_rot[0] - p_curr[0]) + (p_rot[1] - p_curr[1])
    return np.average([zX, zY], weights=[abs(p_rot[0] - p_curr[0]), abs(p_rot[1] - p_curr[1])])
