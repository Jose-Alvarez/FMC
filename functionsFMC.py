# Defining functions for FMC
#
# FMC, Focal Mechanisms Classification
# Copyright (C) 2015  Jose A. Alvarez-Gomez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
# Some of this functions are python adaptations from the
# Gasperini and Vannucci (2003) FORTRAN subroutines:
# Gasperini P. and Vannucci G., FPSPACK: a package of simple Fortran subroutines
# to manage earthquake focal mechanism data, Computers & Geosciences (2003)
#
# Version 1.01
# Version 1.1
# 	Including Hierarchical clustering
# Version 1.2
#	Including slip sense and inmersion
# Version 1.6
# including functions for Hudson source type diamond diagram
# Version 1.7
# include functions to work with P, T orientations as input
# Version 1.8
# Include isotropic component ratio
from numpy import diff, zeros, asarray, sin, cos, sqrt, dot, deg2rad, rad2deg, arccos, arcsin, arctan2, mod, where, linalg, trace, divide
import scipy.cluster.hierarchy as hac


def norm(wax, way, waz):
    """This function Computes Euclidean norm and normalized components of a vector."""
    a = asarray((wax, way, waz))
    anorm = sqrt(dot(a, a.conj()))
    if anorm == 0:
        ax = 0
        ay = 0
        az = 0
    else:
        ax = wax / anorm
        ay = way / anorm
        az = waz / anorm
    return ax, ay, az


def ca2ax(wax, way, waz):
    """This function translates cartesian components to orientation"""
    
    (ax, ay, az) = norm(wax, way, waz)
    if az < 0:
        ax = -ax
        ay = -ay
        az = -az
    if ay != 0 or ax != 0:
        trend = rad2deg(arctan2(ay, ax))
    else:
        trend = 0
    trend = mod(trend + 360, 360)
    plunge = rad2deg(arcsin(az))
    return trend, plunge

def ax2ca(trend, plunge):
    """This function translates orientation to cartesian components"""
    
    ax = cos(deg2rad(plunge))*cos(deg2rad(trend))
    ay = cos(deg2rad(plunge))*sin(deg2rad(trend))
    az = sin(deg2rad(plunge))
    
    return ax, ay, az


def nd2pl(wanx, wany, wanz, wdx, wdy, wdz):
    """This function computes plane orientation from outward normal and slip vectors"""
    (anX, anY, anZ) = norm(wanx, wany, wanz)
    (dx, dy, dz) = norm(wdx, wdy, wdz)

    if anZ > 0:
        anX = -anX
        anY = -anY
        anZ = -anZ
        dx = -dx
        dy = -dy
        dz = -dz
    if anZ == -1:
        wdelta = 0
        wphi = 0
        walam = arctan2(-dy, dx)
    else:
        wdelta = arccos(-anZ)
        wphi = arctan2(-anX, anY)
        walam = arctan2(-dz / sin(wdelta), dx * cos(wphi) + dy * sin(wphi))

    phi = rad2deg(wphi)
    delta = rad2deg(wdelta)
    alam = rad2deg(walam)
    phi = mod(phi + 360, 360)
    dipdir = phi + 90
    dipdir = mod(dipdir + 360, 360)
    return phi, delta, alam, dipdir


def pl2nd(strike, dip, rake):
    """ compute Cartesian components of outward normal and slip vectors from strike, dip and rake
    strike         strike angle in degrees (INPUT)
dip            dip angle in degrees (INPUT)
rake           rake angle in degrees (INPUT)
anx,any,anz    components of fault plane outward normal vector in the
               Aki-Richards Cartesian coordinate system (OUTPUT)
dx,dy,dz       components of slip versor in the Aki-Richards
               Cartesian coordinate system (OUTPUT)"""

    wstrik = deg2rad(strike)
    wdip = deg2rad(dip)
    wrake = deg2rad(rake)

    anX = -sin(wdip) * sin(wstrik)
    anY = sin(wdip) * cos(wstrik)
    anZ = -cos(wdip)
    dx = cos(wrake) * cos(wstrik) + cos(wdip) * sin(wrake) * sin(wstrik)
    dy = cos(wrake) * sin(wstrik) - cos(wdip) * sin(wrake) * cos(wstrik)
    dz = -sin(wdip) * sin(wrake)

    return anX, anY, anZ, dx, dy, dz


def pl2pl(strika, dipa, rakea):
    """Compute one nodal plane from the other."""

    anX, anY, anZ, dx, dy, dz = pl2nd(strika, dipa, rakea)
    strikb, dipb, rakeb, dipdirb = nd2pl(dx, dy, dz, anX, anY, anZ)

    return strikb, dipb, rakeb, dipdirb


def nd2pt(wanx, wany, wanz, wdx, wdy, wdz):
    """compute Cartesian component of P, T and B axes from outward normal and slip vectors."""
    (anX, anY, anZ) = norm(wanx, wany, wanz)
    (dx, dy, dz) = norm(wdx, wdy, wdz)
    px = anX - dx
    py = anY - dy
    pz = anZ - dz
    (px, py, pz) = norm(px, py, pz)
    if pz < 0:
        px = -px
        py = -py
        pz = -pz
    tx = anX + dx
    ty = anY + dy
    tz = anZ + dz
    (tx, ty, tz) = norm(tx, ty, tz)
    if tz < 0:
        tx = -tx
        ty = -ty
        tz = -tz
    bx = py * tz - pz * ty
    by = pz * tx - px * tz
    bz = px * ty - py * tx
    if bz < 0:
        bx = -bx
        by = -by
        bz = -bz

    return px, py, pz, tx, ty, tz, bx, by, bz

def pt2nd(wpx, wpy, wpz, wtx, wty, wtz):
    """compute outward normal and slip vectors from cartesian component of P and T axes."""
    
    (px, py, pz) = norm(wpx, wpy, wpz)
    if pz < 0:
        px = -px
        py = -py
        pz = -pz
    
    (tx, ty, tz) = norm(wtx, wty, wtz)
    if tz < 0:
        tx = -tx
        ty = -ty
        tz = -tz
        
    anX = tx + px
    anY = ty + py
    anZ = tz + pz
    
    (anX, anY, anZ) = norm(anX, anY, anZ)

    dx = tx - px
    dy = ty - py
    dz = tz - pz
    (dx, dy, dz) = norm(dx, dy, dz)
    
    if anZ < 0:
        anX = -anX
        anY = -anY
        anZ = -anZ
        dx = -dx
        dy = -dy
        dz = -dz

    return anX, anY, anZ, dx, dy, dz

def pt2pl(trendp, plungp, trendt, plungt):
    """compute strike dip and rake (and dip direction) of two nodal planes from trend and plung of P and T axes"""
    
    (px, py, pz) = ax2ca(trendp,plungp)
    (tx, ty, tz) = ax2ca(trendt,plungt)
    
    (anX, anY, anZ, dx, dy, dz) = pt2nd(px, py, pz, tx, ty, tz)
    (strika, dipa, rakea, dipdira) = nd2pl(anX, anY, anZ, dx, dy, dz)
    (strikb, dipb, rakeb, dipdirb) = nd2pl(dx, dy, dz, anX, anY, anZ)

    return strika, dipa, rakea, dipdira, strikb, dipb, rakeb, dipdirb

def nd2ar(anX, anY, anZ, dx, dy, dz, am0):
    """Compute tensor components from outward normal and slip vectors."""

    wanx, wany, wanz = norm(anX, anY, anZ)
    wdx, wdy, wdz = norm(dx, dy, dz)

    if am0 == 0:
        aam0 = 1.0
    else:
        aam0 = am0

    am = zeros((3, 3))
    am[0][0] = aam0 * 2.0 * wdx * wanx
    am[0][1] = aam0 * (wdx * wany + wdy * wanx)
    am[1][0] = am[0][1]
    am[0][2] = aam0 * (wdx * wanz + wdz * wanx)
    am[2][0] = am[0][2]
    am[1][1] = aam0 * 2.0 * wdy * wany
    am[1][2] = aam0 * (wdy * wanz + wdz * wany)
    am[2][1] = am[1][2]
    am[2][2] = aam0 * 2.0 * wdz * wanz

    return am


def ar2ha(am):
    """Translates tensor components between cartesian and Harvard convention."""

    amo = zeros((3, 3))
    amo[0][0] = am[0][0]
    amo[0][1] = -am[0][1]
    amo[0][2] = am[0][2]
    amo[1][0] = -am[1][0]
    amo[1][1] = am[1][1]
    amo[1][2] = -am[1][2]
    amo[2][0] = am[2][0]
    amo[2][1] = -am[2][1]
    amo[2][2] = am[2][2]

    return amo


def slipinm(strike, dip, rake):
    """Computes slip vector orientation from a plane orientation."""

    a = cos(deg2rad(rake)) * cos(deg2rad(strike)) + \
        sin(deg2rad(rake)) * cos(deg2rad(dip)) * sin(deg2rad(strike))
    b = -cos(deg2rad(rake)) * sin(deg2rad(strike)) + \
        sin(deg2rad(rake)) * cos(deg2rad(dip)) * cos(deg2rad(strike))
    slip = rad2deg(arctan2(-b, a))
    slip = mod((slip) + 360, 360)
    inmer = rad2deg(arcsin(sin(deg2rad(rake)) * sin(deg2rad(dip))))

    return slip, inmer


def kave(plungt, plungb, plungp):
    """Computes x and y for the Kaverina diagram"""
    
    zt = sin(deg2rad(plungt))
    zb = sin(deg2rad(plungb))
    zp = sin(deg2rad(plungp))
    L = 2 * sin(0.5 * arccos((zt + zb + zp) / sqrt(3)))
    N = sqrt(2 * ((zb - zp)**2 + (zb - zt)**2 + (zt - zp)**2))
    x = sqrt(3) * (L / N) * (zt - zp)
    y = (L / N) * (2 * zb - zp - zt)
    return x, y


def mecclass(plungt, plungb, plungp):
    """Classify the rupture as function of the axes plunges."""


    plunges = asarray((plungp, plungb, plungt))
    P = plunges[0]
    B = plunges[1]
    T = plunges[2]
    maxplung, axis = plunges.max(0), plunges.argmax(0)
    if maxplung >= 67.5:
        if axis == 0:  # P max
            clase = 'N'  # normal faulting
        elif axis == 1:  # B max
            clase = 'SS'  # strike-slip faulting
        elif axis == 2:  # T max
            clase = 'R'  # reverse faulting
    else:
        if axis == 0:  # P max
            if B > T:
                clase = 'N-SS'  # normal - strike-slip faulting
            else:
                clase = 'N'  # normal faulting
        if axis == 1:  # B max
            if P > T:
                clase = 'SS-N'  # strike-slip - normal faulting
            else:
                clase = 'SS-R'  # strike-slip - reverse faulting
        if axis == 2:  # T max
            if B > P:
                clase = 'R-SS'  # reverse - strike-slip faulting
            else:
                clase = 'R'  # reverse faulting
    return clase


def moment(am):
    """Computes scalar seismic moment, fclvd, deviatoric components, iso component and ratio, eigenvectors, and position on the Hudson diagram"""

    # To avoid problems with cosines
    ceros = where(am == 0)
    am[ceros] = 0.000001

    # Eigenvalues and Eigenvectors
    val, vect = linalg.eig(am)
    # Ordering of eigenvalues and eigenvectors (increasing eigenvalues)
    idx = val.argsort()
    val = val[idx]
    vect = vect[:, idx]

    # Tensor isotropic component
    e = trace(am) / 3
    dval = val - e
    iso = e

    # fclvd, seismic moment and Mw
    fclvd = (abs(val[1] / (max((abs(val[0])), (abs(val[2])))))) # from Frohlich and Apperson, 1992
#    am0 = (abs(val[0]) + abs(val[2])) / 2  # From Dziewonski et al., 1981
    am0 = sqrt((val[0]**2 + val[1]**2 + val[2]**2) / 2)  # From Silver and Jordan, 1982
    fiso = iso/am0

    # u & v position in Hudson et al. (1989) skewed diamond from Vavrycuk (2014)
    maxiM = max(abs(val[0]),abs(val[1]),abs(val[2]))
    Ms = divide(val,maxiM)
    u = (-(2/3))*(Ms[2]+Ms[0]-2*Ms[1])
    v = (1/3)*(Ms[0]+Ms[1]+Ms[2])

    return am0, fclvd, dval, vect, iso, u, v, fiso


def HC(data, meth, metr, num_clust):
    """# Mahalanobis Hierarchycal Clustering
    # 	data: 	the set of variables used to perform the clustering analysis
    #	method:	method to perform the HCA [single(default), complete, average, weighted, average, centroid, median, ward]
    #	metric: the metric to perform the HCA [euclidean(default), mahalanobis]
    # num_clust:      predefined number of clusters, if not present then it is
    # automatically computed with "diff"."""

    li = hac.linkage(data, method=meth, metric=metr)
    if num_clust == 0:
        knee = diff(li[::-1, 2], 2)
        num_clust = knee.argmax() + 2
        clustID = hac.fcluster(li, num_clust, 'maxclust')
    else:
        clustID = hac.fcluster(li, num_clust, 'maxclust')
    return clustID
