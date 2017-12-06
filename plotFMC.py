# Plotting functions for FMC
#
# FMC, Focal Mechanisms Classification
# Copyright (C) 2013  Jose A. Alvarez-Gomez
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

import matplotlib.pyplot as plt
from numpy import zeros, sqrt, arcsin, pi
from functionsFMC import kave

plt.rc('pdf', fonttype=3)

# Modified to show also the P axis labels (05/10/2017)
def baseplot(plotname):
	# border	
	fig=plt.figure()
	plt.axes().set_aspect('equal')
	
	X=zeros((1,101))
	Y=zeros((1,101))
	for a in range (0,101):
		P=arcsin(sqrt(a/100.0))/(pi/180)
		B=0.0
		T=arcsin(sqrt(1-(a/100.0)))/(pi/180)
		X[0][a],Y[0][a]=kave(T,B,P)
		
	tickx,ticky=kave(range(90,-1,-10),zeros((1,10)),range(0,91,10))
	plt.plot(X[0],Y[0],color='black',linewidth=2)
	plt.scatter(tickx,ticky,marker=3,c='black',linewidth=2)
	for i in range(0,10):
		plt.text(tickx[0][i],ticky[0][i]-0.04,i*10,fontsize=9,verticalalignment='top')
	plt.text(0,-0.75,'P axis plunge',fontsize=9,horizontalalignment='center')
		
	X=zeros((1,101))
	Y=zeros((1,101))
	for a in range (0,101):
		B=arcsin(sqrt(a/100.0))/(pi/180)
		P=0.0
		T=arcsin(sqrt(1-(a/100.0)))/(pi/180)
		X[0][a],Y[0][a]=kave(T,B,P)
	
	tickx,ticky=kave(zeros((1,10)),range(0,91,10),range(90,-1,-10))
	plt.plot(X[0],Y[0],color='black',linewidth=2)
	plt.scatter(tickx,ticky,marker=0,c='black',linewidth=2)
	for i in range(0,10):
		plt.text(tickx[0][i]-0.04,ticky[0][i],i*10,fontsize=9,horizontalalignment='right')
	plt.text(-0.65,0.5,'B axis plunge',fontsize=9,horizontalalignment='center',rotation=60)

	X=zeros((1,101))
	Y=zeros((1,101))
	for a in range (0,101):
		T=arcsin(sqrt(a/100.0))/(pi/180)
		B=0.0
		P=arcsin(sqrt(1-(a/100.0)))/(pi/180)
		X[0][a],Y[0][a]=kave(T,B,P)
	
	tickx,ticky=kave(range(0,91,10),range(90,-1,-10),zeros((1,10)))
	plt.plot(X[0],Y[0],color='black',linewidth=2)
	plt.scatter(tickx+0.025,ticky,marker=0,c='black',linewidth=2)
	for i in range(0,10):
		plt.text(tickx[0][i]+0.04,ticky[0][i],i*10,fontsize=9,horizontalalignment='left')
	plt.text(0.65,0.5,'T axis plunge',fontsize=9,horizontalalignment='center',rotation=-60)

	
	X=zeros((1,101))
	Y=zeros((1,101))
	for a in range (0,101):
		P=arcsin(sqrt(a/100.0))/(pi/180)
		T=0.0
		B=arcsin(sqrt(1-(a/100.0)))/(pi/180)
		X[0][a],Y[0][a]=kave(T,B,P)
	
	plt.plot(X[0],Y[0],color='black',linewidth=2)
	
	# inner lines
	# class fields
	X=zeros((1,51))
	Y=zeros((1,51))
	for a in range(0,51):
		B=arcsin(sqrt((a/50.0)*0.14645))/(pi/180)
		T=67.5
		P=arcsin(sqrt((1-(a/50.0))*0.14645))/(pi/180)
		X[0][a],Y[0][a]=kave(T,B,P)
	
	xf=X[0][25]
	yf=Y[0][25]
	plt.plot([0,xf],[0,yf],color='grey',linewidth=1)
	plt.plot(X[0][25:51],Y[0][25:51],color='grey',linewidth=1)
	
	X=zeros((1,51))
	Y=zeros((1,51))
	for a in range(0,51):
		B=arcsin(sqrt((a/50.0)*0.14645))/(pi/180)
		P=67.5
		T=arcsin(sqrt((1-(a/50.0))*0.14645))/(pi/180)
		X[0][a],Y[0][a]=kave(T,B,P)
	
	xf=X[0][25]
	yf=Y[0][25]
	plt.plot([0,xf],[0,yf],color='grey',linewidth=1)
	plt.plot(X[0][25:51],Y[0][25:51],color='grey',linewidth=1)
	
	X=zeros((1,51))
	Y=zeros((1,51))
	for a in range(0,51):
		T=arcsin(sqrt((a/50.0)*0.14645))/(pi/180)
		B=67.5
		P=arcsin(sqrt((1-(a/50.0))*0.14645))/(pi/180)
		X[0][a],Y[0][a]=kave(T,B,P)

	plt.plot(X[0],Y[0],color='grey',linewidth=1)
	
	plt.plot([0,0],[0.555221438,-0.605810893],color='grey',linewidth=1)
	plt.plot([0,0.52481139],[0,0.303],color='grey',linewidth=1)
	plt.plot([0,-0.52481139],[0,0.303],color='grey',linewidth=1)
	# Labels
	plt.text(0,-0.9,plotname,horizontalalignment='center',fontsize=16)
	plt.text(-0.9,-0.5,'Normal',horizontalalignment='right',fontsize=14)
	plt.text(0.9,-0.5,'Reverse',horizontalalignment='left',fontsize=14)
	plt.text(0,1,'Strike-slip',horizontalalignment='center',fontsize=14)
	
	plt.axis('off')
		
	return fig

def circles(X,Y,size,color,plotname,label):
	
	fig=baseplot(plotname)
	sc=plt.scatter(X,Y,s=size,c=color,alpha=0.7,linewidth=1.5)
	cbar=plt.colorbar(sc,shrink=0.5)
	cbar.set_label(label)
	# legend
	plt.scatter (0.3,0.9,s=40,c='white',linewidth=1.5)
	plt.scatter (0.4,0.9,s=50,c='white',linewidth=1.5)
	plt.scatter (0.5,0.9,s=60,c='white',linewidth=1.5)
	plt.scatter (0.6,0.9,s=70,c='white',linewidth=1.5)
	plt.scatter (0.7,0.9,s=80,c='white',linewidth=1.5)
	plt.text (0.3,.95,'4',fontsize=10)
	plt.text (0.4,.95,'5',fontsize=10)
	plt.text (0.5,.95,'6',fontsize=10)
	plt.text (0.6,.95,'7',fontsize=10)
	plt.text (0.7,.95,'8',fontsize=10)
	plt.text (0.75,.95,'Mw',fontsize=10)
	return fig

def annot(X,Y,size,color,plotname,label,annots):

	fig=circles(X,Y,size,color,plotname,label)
	for i, txt in enumerate(annots):
    		plt.annotate(txt, (X[i]+0.02,Y[i]+0.02),horizontalalignment='left',verticalalignment='bottom',rotation=30,size='x-small')

	return fig
