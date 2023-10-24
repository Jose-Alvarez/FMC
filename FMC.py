#!/usr/bin/env python3

# FMC, Focal Mechanisms Classification
# Copyright (C) 2015  Jose A. Alvarez-Gomez
#
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
# Version 1.1
#    Including:
#    new output parsing
#    symbol coloring
#    Hierarchical clustering with several methods
#
# Version 1.2
#    Including:
#    Slip sense and inmersion optional output
#
# Version 1.3
#    Including:
#    Adapted to python 3
#   new plot options: labels, colors, grid
#
# Version 1.4
#   Including:
#   Bug correction
#   custom title
#   Isotropic component output
#
# Version 1.5
# Including:
#   Bug correction
#   Warning for symbol filling
#
# Version 1.51
# Including:
#   Correction on genfromtext
#   Adjustment of T and B axes labels
#
# Version 1.6
# Including:
#   Hudson et al. (1989) source-type diagram with plotting options -pd
#
# Version 1.7
# Including:
#  Input of P and T axes from strain tensors obtained with fault slip analysis
#
# Version 1.8
# Including:
#  Isotropic component ratio output
#
# Version 1.9
# Including:
#  Small code adaptation to define symbol plot sizes according to the data input magnitude range    


import sys
import argparse
from argparse import RawTextHelpFormatter, ArgumentParser
from numpy import c_, vstack, array, zeros, asarray, genfromtxt, atleast_2d, shape, log10, array2string, isnan
from functionsFMC import *
from plotFMC import *

# All the command line parser thing.
parser = ArgumentParser(description='Focal mechanism process\
 and classification.\nVersion 1.3', formatter_class=RawTextHelpFormatter)
parser.add_argument('infile', nargs='?')
parser.add_argument('-i', nargs=1, default=['CMT'], choices=['CMT', 'AR', 'P', 'PT'],
                    help='Input file format.\n\
Choose between:\n\
[CMT]: Global CMT for psmeca (GMT) [default]\n\
[AR]: Aki and Richards for psmeca (GMT)\n\
[P]: Old Harvard CMT with both planes for psmeca (GMT)\n\
[PT]: Tensor P and T axes\n')
parser.add_argument(
    '-o', nargs=1, default=['CMT'], choices=['CMT', 'P', 'AR', 'K', 'ALL', 'CUSTOM'],
    help='Output file format.\n\
Choose between:\n\
[CMT]: Global CMT for psmeca (GMT) [default]\n\
[P]: Old Harvard CMT with both planes for psmeca (GMT)\n\
[AR]: Aki and Richards for psmeca (GMT)\n\
[K]: X, Y positions for the Kaverina diagram with Mw, depth, ID and class\n\
[ALL]: A complete format file that outputs all the parameters computed\n\
[CUSTOM]: The outputs fields are given with the option -of [fields].\n\
Type "FMC.py -helpFields" to obtain information on the data fields \n\
that can be used and parsed as comma separated names.\n\
(see details on manual)\n ')
parser.add_argument('-of', nargs='?',
                    help='If present together with -o \'CUSTOM\' FMC will use the fields given as output.\n ')
parser.add_argument('-p', metavar='[PlotFileName.pdf]', nargs='?',
                    help='If present FMC will generate a plot with the DC classification diagram\n\
with the file format specified in the plot file name.\n ')
parser.add_argument('-pd', metavar='[PlotFileName.pdf]', nargs='?',
                    help='If present FMC will generate a plot with the source type classification diagram\n\
with the file format specified in the plot file name.\n ')
parser.add_argument('-pc', nargs='?',
                    help='If present FMC will use the specified parameter to fill the plotted\n\
circles with color in the classification diagram.\n\
Type "FMC.py -helpFields" to obtain information on the data fields\n\
that can be used.\n\
By default FMC uses white circles or, for the clustering, the cluster number.\n ')
parser.add_argument('-pa', nargs='?',
                    help='If present the program will plot labels with the selected parameter on the diagram plot.\n\
Type "FMC.py -helpFields" to obtain information on the data fields that can be used. \n ')
parser.add_argument('-pg', nargs='?',
                    help='If present the program will plot gridlines with the specified angular spacing on the diagram plot. [10 by default] \n ')
parser.add_argument('-pt', nargs='?',
                    help='If present the program will plot a title with the specified text on the diagram plot.\n\
If no text is given, or "-pt" is not set, then the output plot file name (without extension) is used by default.\n\
To omit the title just use the space character as text string -pt " ".\n ')
parser.add_argument(
    '-cm', nargs='?', choices=['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'],
    help='If present FMC will perform a hierarchical clustering analysis\n\
of the focal mechanisms distribution on the Kaverina diagram\n\
with the method specified:\n\
[single]: single/min/nearest\n\
[complete]: complete/max/farthest point\n\
[average]: average/UPGMA\n\
[weighted]: weighted/WPGMA\n\
[centroid]: centroid/UPGMC [default]\n\
[median]: median/WPGMC\n\
[ward]: Ward\'s\n\n\
Methods centroid, median and ward are correctly defined only if "euclidean" metric is used.\n ')
parser.add_argument(
    '-ce', nargs='?', choices=['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
                               'euclidean', 'hamming', 'jaccard', 'mahalanobis', 'minkowski', 'seuclidean', 'sqeuclidean'],
    help='If present FMC will perform a hierarchical clustering analysis of the focal\n\
mechanisms distribution on the Kaverina diagram with the metric specified.\n\
The algorithm uses the metrics available in the scipy.spatial.distance.pdist\n\
function which are listed below. By default FMC uses euclidean. \n\n\
The distance function can be braycurtis, canberra, chebyshev, cityblock, correlation,\n\
cosine, euclidean, hamming, jaccard, mahalanobis, minkowski, seuclidean, sqeuclidean.\n\n\
Please check the adequacy of your choice.\n\
As a rule of thumb if the parameters used to perform the clustering are all in\n\
the same units and equivalent magnitudes "euclidean" is a good choice.\n\
If you are using parameters in different units and magnitudes "mahalanobis" should work.\n ')
parser.add_argument('-cn', nargs='?',
                    help='If present FMC will perform a hierarchical clustering analysis\n\
of the focal mechanisms distribution on the Kaverina diagram\n\
with the number of clusters specified.\n\
if 0 [default] the minimum number of clusters is computed automatically.\n ')
parser.add_argument('-ci', nargs='?',
                    help='If present FMC will use the data given to perform the \n\
hierarchical clustering analysis of the focal mechanisms \n\
instead of the position on the Kaverina diagram, \n\
e.g. "-ci lon,lat" in order to perform a spatial clustering. \n\
Type "FMC.py -helpFields" to obtain information on the data fields \n\
that can be used and parsed as comma separated names.\n ')
parser.add_argument('-v', action='count',
                    help='If present the program will show additional processing information.\n ')
parser.add_argument('-helpFields', action='count',
                    help='If present the program will show information on the different\n\
parameters used or generated by FMC and will exit.\n ')

args = parser.parse_args()
args.outfile = sys.stdout

if args.helpFields is not None:
    sys.stderr.write("Parameters used or generated by FMC: \n\n\
lon = longitude \n\
lat = latitude \n\
dep = depth \n\
mrr = mrr centroid moment tensor component \n\
mtt = mtt centroid moment tensor component \n\
mff = mff centroid moment tensor component \n\
mrt = mrt centroid moment tensor component \n\
mrf = mrf centroid moment tensor component \n\
mtf = mtf centroid moment tensor component \n\
mant = mantissa of the seismic moment tensor \n\
expo = exponent of the seismic moment tensor \n\
Mo = Scalar seismic moment \n\
Mw = Moment (or Kanamori) magnitude \n\
strA = Strike of nodal plane A \n\
dipA = Dip of nodal plane A \n\
rakeA = Rake of nodal plane A \n\
strB = Strike of nodal plane B \n\
dipB = Dip of nodal plane B \n\
rakeB = Rake of nodal plane B \n\
slipA = Slip trend of plane A \n\
plungA = Plunge of slip vector of plane A \n\
slipB = Slip trend of plane B \n\
plungB = Plunge of slip vector of plane B \n\
trendp = Trend of P axis \n\
plungp = Plunge of P axis \n\
trendb = Trend of B axis \n\
plungb = Plunge of B axis \n\
trendt = Trend of T axis \n\
plungt = Plunge of T axis \n\
fclvd = Compensated linear vector dipole ratio \n\
iso = Isotropic component of the Moment Tensor \n\
fiso = Isotropic component ratio \n\
u_Hudson = u position on the Hudson diagram \n\
v_Hudson = v position on the Hudson diagram \n\
x_kav = x position on the Kaverina diagram \n\
y_kav = y position on the Kaverina diagram \n\
ID = ID of the event \n\
clas = focal mechanism rupture type \n\
posX = X plotting position for GMT psmeca \n\
posY = Y plotting position for GMT psmeca \n\
clustID = ID number of cluster \n\
data1 = variable to represent any quantity to use with PT axes input \n\n")
    sys.exit(1)

if args.infile:
    if args.v is not None:
        sys.stderr.write(
            ''.join(
                'Working on input file ' +
                args.infile +
                '\n'))

    open(args.infile).read()
elif not sys.stdin.isatty():
    if args.v:
        sys.stderr.write('Working on standard input.\n')

    parser.add_argument(
        'infile',
        nargs='*',
     type=argparse.FileType('r'),
     default=sys.stdin)
    args = parser.parse_args()
    args.outfile = sys.stdout

else:
    parser.print_help()
    sys.exit(1)

# check the python version to different genfromtext syntax

if sys.version_info[0] == 2:
    data = genfromtxt(args.infile, dtype=None)

elif sys.version_info[0] == 3:
    data = genfromtxt(args.infile, dtype=None, encoding=None)

n_events = data.size
if n_events == 1:
    data = atleast_2d(data)[0]
fields = shape(data.dtype.names)[0]

# Output data array generation
lon_all = zeros((n_events, 1))
lat_all = zeros((n_events, 1))
dep_all = zeros((n_events, 1))
mrr_all = zeros((n_events, 1))
mtt_all = zeros((n_events, 1))
mff_all = zeros((n_events, 1))
mrt_all = zeros((n_events, 1))
mrf_all = zeros((n_events, 1))
mtf_all = zeros((n_events, 1))
mant_all = zeros((n_events, 1))
expo_all = zeros((n_events, 1))
Mo_all = zeros((n_events, 1))
Mw_all = zeros((n_events, 1))
strA_all = zeros((n_events, 1))
dipA_all = zeros((n_events, 1))
rakeA_all = zeros((n_events, 1))
strB_all = zeros((n_events, 1))
dipB_all = zeros((n_events, 1))
rakeB_all = zeros((n_events, 1))
slipA_all = zeros((n_events, 1))
plungA_all = zeros((n_events, 1))
slipB_all = zeros((n_events, 1))
plungB_all = zeros((n_events, 1))
trendp_all = zeros((n_events, 1))
plungp_all = zeros((n_events, 1))
trendb_all = zeros((n_events, 1))
plungb_all = zeros((n_events, 1))
trendt_all = zeros((n_events, 1))
plungt_all = zeros((n_events, 1))
fclvd_all = zeros((n_events, 1))
iso_all = zeros((n_events, 1))
fiso_all = zeros((n_events, 1))
u_Hudson_all = zeros((n_events, 1))
v_Hudson_all = zeros((n_events, 1))
x_kav_all = zeros((n_events, 1))
y_kav_all = zeros((n_events, 1))
ID_all = [None] * n_events
#ID_all = zeros((n_events, 1),dtype="int8")
clas_all = [None] * n_events
posX_all = [None] * n_events
posY_all = [None] * n_events
clustID_all = [None] * n_events
data1_all = zeros((n_events, 1))

# false clustID to initialize the variable
clustID = 0

for row in range(n_events):
    if args.i[0] == 'CMT':
        if fields != 13:
            sys.stderr.write(
                "ERROR - Incorrect number of columns (should be 13). - Program aborted")
            sys.exit(1)
        else:
            if args.v is not None:
                sys.stderr.write(
                    ''.join(
                        '\rProcessing ' + str(
                            row + 1) + '/' + str(
                                n_events) + ' focal mechanisms.'))

        lon = data[row][0]
        lat = data[row][1]
        dep = data[row][2]
        posX = data[row][10]
        posY = data[row][11]
        ID = data[row][12]

        # tensor matrix building
        expo = (data[row][9] * 1.0)
        mrr = data[row][3] * 10**expo
        mtt = data[row][4] * 10**expo
        mff = data[row][5] * 10**expo
        mrt = data[row][6] * 10**expo
        mrf = data[row][7] * 10**expo
        mtf = data[row][8] * 10**expo
        am = asarray(([mtt, -mtf, mrt], [-mtf, mff, -mrf], [mrt, -mrf, mrr]))

        # scalar moment and fclvd
        Mo, fclvd, val, vect, iso, u_Hudson, v_Hudson, fiso = moment(am)
        Mw = ((2.0 / 3.0) * log10(Mo)) - 10.733333
        mant_exp = ("%e" % Mo).split('e')
        mant = mant_exp[0]
        expo = mant_exp[1].strip('+')

        # Axis vectors
        px = vect[0, 0]
        py = vect[1, 0]
        pz = vect[2, 0]
        tx = vect[0, 2]
        ty = vect[1, 2]
        tz = vect[2, 2]
        bx = vect[0, 1]
        by = vect[1, 1]
        bz = vect[2, 1]

        # Axis trend and plunge
        trendp, plungp = ca2ax(px, py, pz)
        trendt, plungt = ca2ax(tx, ty, tz)
        trendb, plungb = ca2ax(bx, by, bz)

        # transforming axis reference
        px, py, pz = norm(px, py, pz)
        if pz < 0:
            px = -px
            py = -py
            pz = -pz
        tx, ty, tz = norm(tx, ty, tz)
        if tz < 0:
            tx = -tx
            ty = -ty
            tz = -tz
        anX = tx + px
        anY = ty + py
        anZ = tz + pz
        anX, anY, anZ = norm(anX, anY, anZ)
        dx = tx - px
        dy = ty - py
        dz = tz - pz
        dx, dy, dz = norm(dx, dy, dz)
        if anZ > 0:
            anX = -anX
            anY = -anY
            anZ = -anZ
            dx = -dx
            dy = -dy
            dz = -dz

        # Obtaining geometry of planes
        strA, dipA, rakeA, dipdir1 = nd2pl(anX, anY, anZ, dx, dy, dz)
        strB, dipB, rakeB, dipdir2 = nd2pl(dx, dy, dz, anX, anY, anZ)

        # Obtaining slip vectors
        slipA, plungA = slipinm(strA, dipA, rakeA)
        slipB, plungB = slipinm(strB, dipB, rakeB)

        # x, y Kaverina diagram
        x_kav, y_kav = kave(plungt, plungb, plungp)

        # Focal mechanism classification Alvarez-Gomez, 2009.
        clas = mecclass(plungt, plungb, plungp)
        
        data1=0

    elif args.i[0] == 'AR':
        if fields != 10:
            sys.stderr.write(
                "ERROR - Incorrect number of columns (should be 10). - Program aborted")
            sys.exit(1)
        else:
            if args.v is not None:
                sys.stderr.write(
                    ''.join(
                        '\rProcessing ' + str(
                            row + 1) + '/' + str(
                                n_events) + ' focal mechanisms.'))

        lon = data[row][0]
        lat = data[row][1]
        dep = data[row][2]
        posX = data[row][7]
        posY = data[row][8]
        ID = data[row][9]

        strA = (data[row][3])
        dipA = (data[row][4])
        rakeA = (data[row][5])
        Mw = (data[row][6])
        Mo = 10**(1.5 * (Mw + 10.7333333))
        mant_exp = ("%e" % Mo).split('e')
        mant = mant_exp[0]
        expo = mant_exp[1].strip('+')

        anX, anY, anZ, dx, dy, dz = pl2nd(strA, dipA, rakeA)
        px, py, pz, tx, ty, tz, bx, by, bz = nd2pt(anX, anY, anZ, dx, dy, dz)
        strB, dipB, rakeB, dipdir2 = pl2pl(strA, dipA, rakeA)

        slipA, plungA = slipinm(strA, dipA, rakeA)
        slipB, plungB = slipinm(strB, dipB, rakeB)

        trendp, plungp = ca2ax(px, py, pz)
        trendt, plungt = ca2ax(tx, ty, tz)
        trendb, plungb = ca2ax(bx, by, bz)

        # moment tensor from P and T
        am = nd2ar(anX, anY, anZ, dx, dy, dz, Mo)
        am = ar2ha(am)
        mrr = am[2][2]
        mff = am[1][1]
        mtt = am[0][0]
        mrf = am[1][2]
        mrt = am[0][2]
        mtf = am[0][1]

        # scalar moment and fclvd
        Mo, fclvd, val, vect, iso, u_Hudson, v_Hudson, fiso = moment(am)

        # x, y Kaverina diagram
        x_kav, y_kav = kave(plungt, plungb, plungp)

        # Focal mechanism classification Alvarez-Gomez, 2009.
        clas = mecclass(plungt, plungb, plungp)
        data1=0

    elif args.i[0] == 'P':
        if fields != 14:
            sys.stderr.write(
                "ERROR - Incorrect number of columns (should be 14). - Program aborted")
            sys.exit(1)
        else:
            if args.v is not None:
                sys.stderr.write(
                    ''.join(
                        '\rProcessing ' + str(
                            row + 1) + '/' + str(
                                n_events) + ' focal mechanisms.'))

        lon = data[row][0]
        lat = data[row][1]
        dep = data[row][2]
        posX = data[row][11]
        posY = data[row][12]
        ID = data[row][13]

        strA = (data[row][3])
        dipA = (data[row][4])
        rakeA = (data[row][5])
        strB = (data[row][6])
        dipB = (data[row][7])
        rakeB = (data[row][8])

        mant = (data[row][9] * 1.0)
        expo = (data[row][10] * 1.0)
        Mo = mant * 10**expo
        Mw = ((2.0 / 3.0) * log10(Mo)) - 10.733333

        anX, anY, anZ, dx, dy, dz = pl2nd(strA, dipA, rakeA)
        px, py, pz, tx, ty, tz, bx, by, bz = nd2pt(anX, anY, anZ, dx, dy, dz)

        slipA, plungA = slipinm(strA, dipA, rakeA)
        slipB, plungB = slipinm(strB, dipB, rakeB)

        trendp, plungp = ca2ax(px, py, pz)
        trendt, plungt = ca2ax(tx, ty, tz)
        trendb, plungb = ca2ax(bx, by, bz)

        # moment tensor from P and T
        am = nd2ar(anX, anY, anZ, dx, dy, dz, Mo)
        am = ar2ha(am)
        mrr = am[2][2]
        mff = am[1][1]
        mtt = am[0][0]
        mrf = am[1][2]
        mrt = am[0][2]
        mtf = am[0][1]

        # scalar moment and fclvd
        Mo, fclvd, val, vect, iso, u_Hudson, v_Hudson, fiso = moment(am)

        # x, y Kaverina diagram
        x_kav, y_kav = kave(plungt, plungb, plungp)

        # Focal mechanism classification Alvarez-Gomez, 2009.
        clas = mecclass(plungt, plungb, plungp)
        data1=0

    elif args.i[0] == 'PT': # to work with tensors obtained from fault slip analysis
        if fields != 10:
            sys.stderr.write(
                "ERROR - Incorrect number of columns (should be 10). - Program aborted")
            sys.exit(1)
        else:
            if args.v is not None:
                sys.stderr.write(
                    ''.join(
                        '\rProcessing ' + str(
                            row + 1) + '/' + str(
                                n_events) + ' focal mechanisms.'))
        lon = data[row][0]
        lat = data[row][1]
        trendp = data[row][2]
        plungp = data[row][3]
        trendt = data[row][4]
        plungt = data[row][5]
        data1 = data[row][6]
        posX = data[row][7]
        posY = data[row][8]
        ID = data[row][9]
        
        strA, dipA, rakeA, dipdirA, strB, dipB, rakeB, dipdirB = pt2pl(trendp, plungp, trendt, plungt)
        
        anX, anY, anZ, dx, dy, dz = pl2nd(strA, dipA, rakeA)
        px, py, pz, tx, ty, tz, bx, by, bz = nd2pt(anX, anY, anZ, dx, dy, dz)

        slipA, plungA = slipinm(strA, dipA, rakeA)
        slipB, plungB = slipinm(strB, dipB, rakeB)
        
        trendb, plungb = ca2ax(bx, by, bz)
        
        # moment tensor from P and T
        Mo = 1E20 # fake scalar moment
        expo = 20 # fake exponent
        mant = 1 # fake mantissa
        Mw = 8 # fake Mw
        dep = 0 # fake depth
        
        am = nd2ar(anX, anY, anZ, dx, dy, dz, Mo)
        am = ar2ha(am)
        mrr = am[2][2]
        mff = am[1][1]
        mtt = am[0][0]
        mrf = am[1][2]
        mrt = am[0][2]
        mtf = am[0][1]
        
        # scalar moment and fclvd
        Mo, fclvd, val, vect, iso, u_Hudson, v_Hudson, fiso = moment(am)

        # x, y Kaverina diagram
        x_kav, y_kav = kave(plungt, plungb, plungp)

        # Focal mechanism classification Alvarez-Gomez, 2009.
        clas = mecclass(plungt, plungb, plungp)

    else:
        sys.stderr.write('Error, input file format should be G or P.')
        sys.exit(1)

    # storing data for the plot
    lon_all[row] = "%g" % (lon)
    lat_all[row] = "%g" % (lat)
    dep_all[row] = dep
    mrr_all[row] = "%g" % (mrr / (10**(int(expo))))
    mtt_all[row] = "%g" % (mtt / (10**(int(expo))))
    mff_all[row] = "%g" % (mff / (10**(int(expo))))
    mrt_all[row] = "%g" % (mrt / (10**(int(expo))))
    mrf_all[row] = "%g" % (mrf / (10**(int(expo))))
    mtf_all[row] = "%g" % (mtf / (10**(int(expo))))
    mant_all[row] = mant
    expo_all[row] = expo
    Mo_all[row] = "%g" % (Mo)
    Mw_all[row] = "%.1f" % (Mw)
    strA_all[row] = "%g" % (strA)
    dipA_all[row] = "%g" % (dipA)
    rakeA_all[row] = "%g" % (rakeA)
    strB_all[row] = "%g" % (strB)
    dipB_all[row] = "%g" % (dipB)
    rakeB_all[row] = "%g" % (rakeB)
    slipA_all[row] = "%g" % (slipA)
    plungA_all[row] = "%g" % (plungA)
    slipB_all[row] = "%g" % (slipB)
    plungB_all[row] = "%g" % (plungB)
    trendp_all[row] = "%g" % (trendp)
    plungp_all[row] = "%g" % (plungp)
    trendb_all[row] = "%g" % (trendb)
    plungb_all[row] = "%g" % (plungb)
    trendt_all[row] = "%g" % (trendt)
    plungt_all[row] = "%g" % (plungt)
    fclvd_all[row] = "%g" % (fclvd)
    iso_all[row] = "%g" % (iso)
    fiso_all[row] = "%g" % (fiso)
    u_Hudson_all[row] = "%g" % (u_Hudson)
    v_Hudson_all[row] = "%g" % (v_Hudson)
    x_kav_all[row] = "%g" % (x_kav)
    y_kav_all[row] = "%g" % (y_kav)
    ID_all[row] = ID
#    ID_all[row] = "%g" % (ID)
    clas_all[row] = clas
    posX_all[row] = posX
    posY_all[row] = posY
    clustID_all[row] = clustID
    data1_all[row] = data1


    r = row + 1

lonH = vstack(((['Longitude']), (array(lon_all, dtype=object))))
latH = vstack(((['Latitude']), (array(lat_all, dtype=object))))
depH = vstack(((['Depth_(km)']), (array(dep_all, dtype=object))))
mrrH = vstack(((['mrr']), (array(mrr_all, dtype=object))))
mttH = vstack(((['mtt']), (array(mtt_all, dtype=object))))
mffH = vstack(((['mff']), (array(mff_all, dtype=object))))
mrtH = vstack(((['mrt']), (array(mrt_all, dtype=object))))
mrfH = vstack(((['mrf']), (array(mrf_all, dtype=object))))
mtfH = vstack(((['mtf']), (array(mtf_all, dtype=object))))
mantH = vstack(
    ((['Seismic_moment_mantissa']), (array(mant_all, dtype=object))))
expoH = vstack(((['Exponent_(dyn-cm)']), (array(expo_all, dtype=object))))
MoH = vstack(((['Seismic_moment_Mo']), (array(Mo_all, dtype=object))))
MwH = vstack(((['Magnitude_Mw']), (array(Mw_all, dtype=object))))
strAH = vstack(((['Strike_A']), (array(strA_all, dtype=object))))
dipAH = vstack(((['Dip_A']), (array(dipA_all, dtype=object))))
rakeAH = vstack(((['Rake_A']), (array(rakeA_all, dtype=object))))
strBH = vstack(((['Strike_B']), (array(strB_all, dtype=object))))
dipBH = vstack(((['Dip_B']), (array(dipB_all, dtype=object))))
rakeBH = vstack(((['Rake_B']), (array(rakeB_all, dtype=object))))
slipAH = vstack(((['Slip_trend_A']), (array(slipA_all, dtype=object))))
plungAH = vstack(((['Slip_plunge_A']), (array(plungA_all, dtype=object))))
slipBH = vstack(((['Slip_trend_B']), (array(slipB_all, dtype=object))))
plungBH = vstack(((['Slip_plunge_B']), (array(plungB_all, dtype=object))))
trendpH = vstack(((['Trend_P']), (array(trendp_all, dtype=object))))
plungpH = vstack(((['Plunge_P']), (array(plungp_all, dtype=object))))
trendbH = vstack(((['Trend_B']), (array(trendb_all, dtype=object))))
plungbH = vstack(((['Plunge_B']), (array(plungb_all, dtype=object))))
trendtH = vstack(((['Trend_T']), (array(trendt_all, dtype=object))))
plungtH = vstack(((['Plunge_T']), (array(plungt_all, dtype=object))))
fclvdH = vstack(((['fclvd']), (array(fclvd_all, dtype=object))))
isoH = vstack(((['Isotropic']), (array(iso_all, dtype=object))))
fisoH = vstack(((['Iso_ratio']), (array(fiso_all, dtype=object))))
u_HudsonH = vstack(((['u_Hudson']), (array(u_Hudson_all, dtype=object))))
v_HudsonH = vstack(((['v_Hudson']), (array(v_Hudson_all, dtype=object))))
x_kavH = vstack(((['X_Kaverina']), (array(x_kav_all, dtype=object))))
y_kavH = vstack(((['Y_Kaverina']), (array(y_kav_all, dtype=object))))
IDH = vstack(((['ID']), (array(ID_all).reshape((n_events, 1)))))
clasH = vstack(((['rupture_type']), (array(clas_all).reshape((n_events, 1)))))
posXH = vstack(
    ((['X_position(GMT)']), (array(posX_all).reshape((n_events, 1)))))
posYH = vstack(
    ((['Y_position(GMT)']), (array(posY_all).reshape((n_events, 1)))))
clustIDH = vstack(((['clustID']), (array(clustID_all).reshape((n_events, 1)))))
data1H = vstack(((['data1']), (array(data1_all).reshape((n_events, 1)))))

dict_all = {
    'lon': lon_all,
     'lat': lat_all,
     'dep': dep_all,
     'mrr': mrr_all,
     'mtt': mtt_all,
     'mff': mff_all,
     'mrt': mrt_all,
     'mrf': mrf_all,
     'mtf': mtf_all,
     'mant': mant_all,
     'expo': expo_all,
     'Mo': Mo_all,
     'Mw': Mw_all,
     'strA': strA_all,
     'dipA': dipA_all,
     'rakeA': rakeA_all,
     'strB': strB_all,
     'dipB': dipB_all,
     'rakeB': rakeB_all,
     'slipA': slipA_all,
     'plungA': plungA_all,
     'slipB': slipB_all,
     'plungB': plungB_all,
     'trendp': trendp_all,
     'plungp': plungp_all,
     'trendb': trendb_all,
     'plungb': plungb_all,
     'trendt': trendt_all,
     'plungt': plungt_all,
     'fclvd': fclvd_all,
     'iso': iso_all,
     'fiso': fiso_all,
     'u_Hudson': u_Hudson_all,
     'v_Hudson': v_Hudson_all,
     'x_kav': x_kav_all,
     'y_kav': y_kav_all,
     'ID': ID_all,
     'clas': clas_all,
     'posX': posX_all,
     'posY': posY_all,
     'clustID': clustID_all,
     'data1': data1_all}

if args.v is not None:
    sys.stderr.write('\n')

if args.cn is None and args.cm is None and args.ce is None and args.ci is None:
    clustering = 'FALSE'
else:
    if args.cm is None:
        method = 'centroid'
    else:
        method = args.cm

    if args.ce is None:
        metric = 'euclidean'
    else:
        metric = args.ce

    if args.cn is None:
        num_clust = 0
    else:
        num_clust = int(args.cn)

    if args.ci:
        if "," in args.ci:
            labels = ('%s' % args.ci).split(",")
            nl = len(labels) - 1
            for l in labels:
                if 'cl_input' in locals():
                    cl_input = c_[cl_input, dict_all[l]]
                else:
                    cl_input = dict_all[l]
        else:
            cl_input = dict_all[args.ci]
    else:
        cl_input = c_[x_kav_all, y_kav_all]

    clustID = HC(cl_input, method, metric, num_clust)
    clustID = (array(clustID).reshape((n_events, 1)))
    clustIDH = vstack(
        ((['Cluster_ID']), (clustID)))
    clustering = 'TRUE'

dict_H = {
    'lon': lonH,
     'lat': latH,
     'dep': depH,
     'mrr': mrrH,
     'mtt': mttH,
     'mff': mffH,
     'mrt': mrtH,
     'mrf': mrfH,
     'mtf': mtfH,
     'mant': mantH,
     'expo': expoH,
     'Mo': MoH,
     'Mw': MwH,
     'strA': strAH,
     'dipA': dipAH,
     'rakeA': rakeAH,
     'strB': strBH,
     'dipB': dipBH,
     'rakeB': rakeBH,
     'slipA': slipAH,
     'plungA': plungAH,
     'slipB': slipBH,
     'plungB': plungBH,
     'trendp': trendpH,
     'plungp': plungpH,
     'trendb': trendbH,
     'plungb': plungbH,
     'trendt': trendtH,
     'plungt': plungtH,
     'fclvd': fclvdH,
     'iso': isoH,
     'fiso': fisoH,
     'u_Hudson': u_HudsonH,
     'v_Hudson': v_HudsonH,
     'x_kav': x_kavH,
     'y_kav': y_kavH,
     'ID': IDH,
     'clas': clasH,
     'posX': posXH,
     'posY': posYH,
     'clustID': clustIDH,
     'data1': data1H}

#~ output
if args.o[0] == 'CMT':
    outdata = c_[
        lonH,
        latH,
     depH,
     mrrH,
     mttH,
     mffH,
     mrtH,
     mrfH,
     mtfH,
     expoH,
     posXH,
     posYH,
     IDH,
     clasH]
    if clustering == 'TRUE':
        outdata = c_[outdata, clustIDH]

elif args.o[0] == 'P':
    outdata = c_[
        lonH,
        latH,
     depH,
     strAH,
     dipAH,
     rakeAH,
     strBH,
     dipBH,
     rakeBH,
     mantH,
     expoH,
     posXH,
     posYH,
     IDH,
     clasH]
    if clustering == 'TRUE':
        outdata = c_[outdata, clustIDH]

elif args.o[0] == 'AR':
    outdata = c_[
        lonH,
        latH,
     depH,
     strAH,
     dipAH,
     rakeAH,
     MwH,
     posXH,
     posYH,
     IDH,
     clasH]
    if clustering == 'TRUE':
        outdata = c_[outdata, clustIDH]

elif args.o[0] == 'K':
    outdata = c_[x_kavH, y_kavH, MwH, depH, IDH, clasH]
    if clustering == 'TRUE':
        outdata = c_[outdata, clustIDH]

elif args.o[0] == 'ALL':
    outdata = c_[
        lonH,
        latH,
     depH,
     mrrH,
     mttH,
     mffH,
     mrtH,
     mrfH,
     mtfH,
     expoH,
     MoH,
     MwH,
     strAH,
     dipAH,
     rakeAH,
     strBH,
     dipBH,
     rakeBH,
     slipAH,
     plungAH,
     slipBH,
     plungBH,
     trendpH,
     plungpH,
     trendbH,
     plungbH,
     trendtH,
     plungtH,
     fclvdH,
     isoH,
     fisoH,
     u_HudsonH,
     v_HudsonH,
     x_kavH,
     y_kavH,
     IDH,
     clasH]
    if clustering == 'TRUE':
        outdata = c_[outdata, clustIDH]

elif args.o[0] == 'CUSTOM':
    if "," in args.of:
        labels = ('%s' % args.of).split(",")
        nl = len(labels) - 1
        for l in labels:
            if 'outdata' in locals():
                outdata = c_[outdata, dict_H[l]]
            else:
                outdata = dict_H[l]
    else:
        outdata = dict_H[args.of]

outdata[0][0] = "#" + outdata[0][0]
args.outfile.write(
    '\n'.join(str(e).strip("[]").replace("'", '').replace('\n', '') for e in outdata))
print ("")

# diagram FMC plot

if args.p:
    if args.pc:
#        if args.pc == 'ID' or args.pc == 'posX' or args.pc == 'posY' or args.pc == 'clas':
        if args.pc == 'posX' or args.pc == 'posY' or args.pc == 'clas':
            sys.stderr.write('\nWarning, to fill the symbols a numeric value is needed.\n')
            color = 'white'
            label = 'nada'
        else:
            color = dict_all[args.pc]
            label = str(dict_H[args.pc][0]).strip(
                "[]").replace("'", '').replace("_", " ")
    else:
        if clustering == 'TRUE':
            color = clustID
#            color = (dict_all['clustID']) # tratando de solventar el problema de colorear con los ID de cluster en python3
            label = 'Clust ID'
        else:
            color = 'white'
            label = 'nada'

    if args.pg:
        gridspacing = int(args.pg)
    else:
        gridspacing = 0

    if args.pt:
        plotname = args.pt
    else:
        plotname = args.p.split('.')[0]

# ----------------------------------

    if args.pa:
        dotlabel = dict_all[args.pa]
        lab_param = dict_H[args.pa][0]
        fig = annot(
            x_kav_all,
            y_kav_all,
            Mw_all,
            color,
            plotname,
            label,
            dotlabel,
            lab_param,
            gridspacing)
    else:
        fig = circles(
            x_kav_all,
            y_kav_all,
            Mw_all,
            color,
            plotname,
            label,
            gridspacing)

    plt.savefig(args.p, dpi=300)
    plt.close()

# source type diagram plot
if args.pd:
    if args.pc:
#        if args.pc == 'ID' or args.pc == 'posX' or args.pc == 'posY' or args.pc == 'clas':
        if args.pc == 'posX' or args.pc == 'posY' or args.pc == 'clas':
            sys.stderr.write('\nWarning, to fill the symbols a numeric value is needed.\n')
            color = 'white'
            label = 'nada'
        else:
            color = dict_all[args.pc]
            label = str(dict_H[args.pc][0]).strip(
                "[]").replace("'", '').replace("_", " ")
    else:
        if clustering == 'TRUE':
            color = clustID
            label = 'Clust ID'
        else:
            color = 'white'
            label = 'nada'
    if args.pt:
        plotname = args.pt
    else:
        plotname = args.pd.split('.')[0]
    if args.pa:
        dotlabel = dict_all[args.pa]
        lab_param = dict_H[args.pa][0]
        fig = diamond_annot(
            u_Hudson_all,
            v_Hudson_all,
            Mw_all,
            color,
            plotname,
            label,
            dotlabel,
            lab_param)
#            gridspacing)
    else:
        fig = diamond_circles(
            u_Hudson_all,
            v_Hudson_all,
            Mw_all,
            color,
            plotname,
            label)
#            gridspacing)

    plt.savefig(args.pd, dpi=300)
    plt.close()
