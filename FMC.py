#!/usr/bin/env python

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
# Version 1.01
#
#

import sys, argparse, os

from numpy import zeros, asarray, sin, cos, sqrt, dot, deg2rad, rad2deg, arccos, arcsin, arctan2, mod, genfromtxt, column_stack, atleast_2d, shape, savetxt, where, linalg, trace, log10
from functionsFMC import *
from plotFMC import *

# All the command line parser thing.
parser = argparse.ArgumentParser(description='Focal mechanism process\
 and classification.')
parser.add_argument('infile', nargs='?') #, type=argparse.FileType('r'), default=sys.stdin
parser.add_argument('-i' ,nargs=1,default=['CMT'],choices=['CMT','AR'], \
help='Input file format. Choose between\n [CMT]: Global CMT for psmeca (GMT) [default]; \
[AR]: Aki and Richards for psmeca (GMT)')
parser.add_argument('-o',nargs=1,default=['CMT'],choices=['CMT','P','AR','K','ALL'], \
help='Output file format. Choose between\n [CMT]: Global CMT for psmeca (GMT) [default];\
[P]: Old Harvard CMT with both planes for psmeca (GMT); \
[AR]: Aki and Richards for psmeca (GMT); \
[K]: X, Y positions for the Kaverina diagram with Mw, depth, ID and class; \
[ALL]: A complete format file that outputs all the parameters computed (see details on manual)')
parser.add_argument('-p',metavar='[PlotFileName.pdf]',nargs='?', \
help='If present FMC will generate a plot with the classification diagram with the format specified in the plot file name.')
parser.add_argument('-v',action='count',\
help='If present the program will show additional processing information.')

args = parser.parse_args()
args.outfile = sys.stdout

if args.infile:
	if args.v > 0:
		sys.stderr.write(''.join('Working on input file '+ args.infile + '\n'))

	open(args.infile).read()
elif not sys.stdin.isatty():
	if args.v:
		sys.stderr.write('Working on standard input.\n')

	parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
	args = parser.parse_args()
	args.outfile = sys.stdout

else:
	parser.print_help()
	sys.exit(1)

data = genfromtxt(args.infile,dtype=None)
n_events = data.size
if n_events == 1:
	data=atleast_2d(data)[0]
fields = shape(data.dtype.names)[0]

# Output data array generation
if args.o[0] == 'CMT':
	outdata = zeros((n_events+1,14)).tolist()
elif args.o[0] == 'P':
	outdata = zeros((n_events+1,15)).tolist()
elif args.o[0] == 'AR':
	outdata = zeros((n_events+1,11)).tolist()
elif args.o[0] == 'K':
	outdata = zeros((n_events+1,6)).tolist()
elif args.o[0] == 'ALL':
	outdata = zeros((n_events+1,29)).tolist()
else:
	sys.stderr.write("ERROR - Incorrect type of output file")
	sys.exit(1)

xplot=zeros((n_events,1))
yplot=zeros((n_events,1))
Mwplot=zeros((n_events,1))
depplot=zeros((n_events,1))

for row in range(n_events):
	if args.i[0] == 'CMT':
		if fields != 13:
			sys.stderr.write("ERROR - Incorrect number of columns (should be 13). - Program aborted")
			sys.exit(1)
		else:
			if args.v > 0:
				sys.stderr.write(''.join('\rProcessing '+str(row+1)+'/'+str(n_events)+' focal mechanisms.'))

		lon=data[row][0]
		lat=data[row][1]
		dep=data[row][2]
		posX=data[row][10]
		posY=data[row][11]
		ID=data[row][12]
		# tensor matrix building
		expo=(data[row][9]*1.0)
		mrr=data[row][3]*10**expo
		mtt=data[row][4]*10**expo
		mff=data[row][5]*10**expo
		mrt=data[row][6]*10**expo
		mrf=data[row][7]*10**expo
		mtf=data[row][8]*10**expo
		am=asarray(([mtt,-mtf,mrt],[-mtf,mff,-mrf],[mrt,-mrf,mrr]))

		# scalar moment and fclvd
		am0, fclvd, val, vect = moment(am)
		Mw=((2.0/3.0)*log10(am0))-10.7
		mant_exp = ("%e" % am0).split('e')
		mant=mant_exp[0]
		expo=mant_exp[1].strip('+')

		# Axis vectors
		px=vect[0,0]
		py=vect[1,0]
		pz=vect[2,0]
		tx=vect[0,2]
		ty=vect[1,2]
		tz=vect[2,2]
		bx=vect[0,1]
		by=vect[1,1]
		bz=vect[2,1]

		# Axis trend and plunge
		trendp,plungp=ca2ax(px,py,pz)
		trendt,plungt=ca2ax(tx,ty,tz)
		trendb,plungb=ca2ax(bx,by,bz)

		# transforming axis reference
		px,py,pz=norm(px,py,pz)
		if pz<0:
			px=-px
			py=-py
			pz=-pz
		tx,ty,tz=norm(tx,ty,tz)
		if tz<0:
			tx=-tx
			ty=-ty
			tz=-tz
		anX=tx+px
		anY=ty+py
		anZ=tz+pz
		anX,anY,anZ=norm(anX,anY,anZ)
		dx=tx-px
		dy=ty-py
		dz=tz-pz
		dx,dy,dz=norm(dx,dy,dz)
		if anZ>0:
			anX=-anX
			anY=-anY
			anZ=-anZ
			dx=-dx
			dy=-dy
			dz=-dz

		# Obtaining geometry of planes
		str1,dip1,rake1,dipdir1=nd2pl(anX,anY,anZ,dx,dy,dz)
		str2,dip2,rake2,dipdir2=nd2pl(dx,dy,dz,anX,anY,anZ)

		# x, y Kaverina diagram
		x_kav,y_kav=kave(plungt,plungb,plungp)

		# Focal mechanism classification Alvarez-Gomez, 2009.
		clase=mecclass(plungt,plungb,plungp)

	elif args.i[0] == 'AR':
		if fields != 10:
			sys.stderr.write("ERROR - Incorrect number of columns (should be 10). - Program aborted")
			sys.exit(1)
		else:
			if args.v > 0:
				sys.stderr.write(''.join('\rProcessing '+str(row+1)+'/'+str(n_events)+' focal mechanisms.'))

		lon=data[row][0]
		lat=data[row][1]
		dep=data[row][2]
		posX=data[row][7]
		posY=data[row][8]
		ID=data[row][9]

		str1=(data[row][3])
		dip1=(data[row][4])
		rake1=(data[row][5])
		Mw=(data[row][6])
		am0=10**(1.5*(Mw+10.7))
		mant_exp = ("%e" % am0).split('e')
		mant=mant_exp[0]
		expo=mant_exp[1].strip('+')

		anX, anY, anZ, dx, dy, dz = pl2nd(str1,dip1,rake1)
		px, py, pz, tx, ty, tz, bx, by, bz = nd2pt(anX,anY,anZ,dx,dy,dz)
		str2,dip2,rake2,dipdir2 = pl2pl(str1,dip1,rake1)

		trendp,plungp=ca2ax(px,py,pz)
		trendt,plungt=ca2ax(tx,ty,tz)
		trendb,plungb=ca2ax(bx,by,bz)

		# moment tensor from P and T
		am=nd2ar(anX,anY,anZ,dx,dy,dz,am0)
		am=ar2ha(am)
		mrr=am[2][2]
		mff=am[1][1]
		mtt=am[0][0]
		mrf=am[1][2]
		mrt=am[0][2]
		mtf=am[0][1]

		# scalar moment and fclvd
		am0, fclvd, val, vect = moment(am)

		# x, y Kaverina diagram
		x_kav,y_kav=kave(plungt,plungb,plungp)

		# Focal mechanism classification Alvarez-Gomez, 2009.
		clase=mecclass(plungt,plungb,plungp)

	else:
		sys.stderr.write('Error, input file format should be G or P.')
		sys.exit(1)
	r=row+1

	# storing data for the plot
	xplot[row] = x_kav
	yplot[row] = y_kav
	Mwplot[row] = Mw
	depplot[row] = row

	if args.o[0] == 'CMT':
		outdata[0]='#Lon, Lat, Depth, mrr, mtt, mff, mrt, mrf, mtf, Exponent_(dyn-cm), X_position, Y_position, ID, Mechanism_type'
		outdata[r][0] = "%g" %(lon)
		outdata[r][1] = "%g" %(lat)
		outdata[r][2] = dep
		outdata[r][3] = "%g" %(mrr/(10**(int(expo))))
		outdata[r][4] = "%g" %(mtt/(10**(int(expo))))
		outdata[r][5] = "%g" %(mff/(10**(int(expo))))
		outdata[r][6] = "%g" %(mrt/(10**(int(expo))))
		outdata[r][7] = "%g" %(mrf/(10**(int(expo))))
		outdata[r][8] = "%g" %(mtf/(10**(int(expo))))
		outdata[r][9] = expo
		outdata[r][10] = posX
		outdata[r][11] = posY
		outdata[r][12] = ID
		outdata[r][13] = clase

	elif args.o[0] == 'P':
		outdata[0]='#Lon, Lat, Depth, Strike_A, Dip_A, Rake_A, Strike_B, Dip_B, Rake_B, Seismic_moment_mantissa, Exponent_(dyn-cm), X_position, Y_position, ID, Mechanism_type'
		outdata[r][0] = "%g" %(lon)
		outdata[r][1] = "%g" %(lat)
		outdata[r][2] = dep
		outdata[r][3] = "%g" %(str1)
		outdata[r][4] = "%g" %(dip1)
		outdata[r][5] = "%g" %(rake1)
		outdata[r][6] = "%g" %(str2)
		outdata[r][7] = "%g" %(dip2)
		outdata[r][8] = "%g" %(rake2)
		outdata[r][9] = mant
		outdata[r][10] = expo
		outdata[r][11] = posX
		outdata[r][12] = posY
		outdata[r][13] = ID
		outdata[r][14] = clase

	elif args.o[0] == 'AR':
		outdata[0]='#Lon, Lat, Depth, Strike_A, Dip_A, Rake_A, Mw, X_position, Y_position, ID, Mechanism_type'
		outdata[r][0] = "%g" %(lon)
		outdata[r][1] = "%g" %(lat)
		outdata[r][2] = dep
		outdata[r][3] = "%g" %(str1)
		outdata[r][4] = "%g" %(dip1)
		outdata[r][5] = "%g" %(rake1)
		outdata[r][6] = "%g" %(Mw)
		outdata[r][7] = posX
		outdata[r][8] = posY
		outdata[r][9] = ID
		outdata[r][10] = clase

	elif args.o[0] == 'K':
		outdata[0]='#X_Kaverina_diagram, Y_Kaverina_diagram, Mw, Depth, ID, Mechanism_type'
		outdata[r][0] = "%g" %(x_kav)
		outdata[r][1] = "%g" %(y_kav)
		outdata[r][2] = "%g" %(Mw)
		outdata[r][3] = dep
		outdata[r][4] = ID
		outdata[r][5] = clase

	elif args.o[0] == 'ALL':
		outdata[0]='#Lon_[1], Lat_[2], Depth_[3], mrr_[4], mtt_[5], mff_[6], mrt_[7], mrf_[8], mtf_[9], Exponent_(dyn-cm)_[10], Scalar_seismic_moment_(dyn-cm)_[11], Mw_[12], Strike_A_[13], Dip_A_[14], Rake_A_[15], Strike_B_[16], Dip_B_[17], Rake_B_[18], P_Trend_[19], P_plunge_[20], B_trend_[21], B_plunge_[22], T_trend_[23], T_plunge_[24], fclvd_[25], X_Kaverina_diagram_[26], Y_Kaverina_diagram_[27], ID_[28], Mechanism_type_[29]'
		outdata[r][0] = "%g" %(lon)
		outdata[r][1] = "%g" %(lat)
		outdata[r][2] = dep
		outdata[r][3] = "%g" %(mrr/(10**int(expo)))
		outdata[r][4] = "%g" %(mtt/(10**int(expo)))
		outdata[r][5] = "%g" %(mff/(10**int(expo)))
		outdata[r][6] = "%g" %(mrt/(10**int(expo)))
		outdata[r][7] = "%g" %(mrf/(10**int(expo)))
		outdata[r][8] = "%g" %(mtf/(10**int(expo)))
		outdata[r][9] = expo
		outdata[r][10] = "%g" %(am0)
		outdata[r][11] = "%g" %(Mw)
		outdata[r][12] = "%g" %(str1)
		outdata[r][13] = "%g" %(dip1)
		outdata[r][14] = "%g" %(rake1)
		outdata[r][15] = "%g" %(str2)
		outdata[r][16] = "%g" %(dip2)
		outdata[r][17] = "%g" %(rake2)
		outdata[r][18] = "%g" %(trendp)
		outdata[r][19] = "%g" %(plungp)
		outdata[r][20] = "%g" %(trendb)
		outdata[r][21] = "%g" %(plungb)
		outdata[r][22] = "%g" %(trendt)
		outdata[r][23] = "%g" %(plungt)
		outdata[r][24] = "%g" %(fclvd)
		outdata[r][25] = "%g" %(x_kav)
		outdata[r][26] = "%g" %(y_kav)
		outdata[r][27] = ID
		outdata[r][28] = clase
if args.v > 0:
	sys.stderr.write('\n')


# diagram FMC plot
# borders

#~ for item in outdata:
#~ print outdata
#~ savetxt(args.outfile,outdata,delimiter=' ')
args.outfile.write('\n'.join(str(str(e)).strip("[]").replace("\'",'').replace(",",'') for e in outdata))
print ""
#~ args.outfile.close()

if args.p==None:
	sys.exit(1)
else:
	fig=circles(xplot,yplot,Mwplot*10,depplot,args.p.split('.')[0])
	plt.savefig(args.p)
	plt.close()
