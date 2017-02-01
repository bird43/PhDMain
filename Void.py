import numpy as np
#import matplotlib.pyplot as plt
#import TestLinePy as lt
import tl1 as tl1
import tl2 as tl2
import tl3 as tl3
import tl4 as tl4
import tl5 as tl5
import tl6 as tl6
import tl7 as tl7
import tl8 as tl8
import tl9 as tl9
from time import time

##offSource = open("MuseumFull1.off", "r")
##print "museum"
#offSource = open("MeshWithGoodHoles.off", "r")
#offSource = open("SHPMesh.off", "r")
#data = offSource.readlines()
##data = offSource.read().split('\n')
#Indexing = data[1]
##PointIndex = np.fromstring(data[1], dtype=int, sep=" ")[0]
##TriangleIndex = np.fromstring(data[1], dtype=int, sep=" ")[1]

##Points = np.zeros([PointIndex,3])
##Triangles = np.zeros([TriangleIndex,3])



##for i in range(2, PointIndex+2):
##    Points[i-2,0] = np.fromstring(data[i], dtype=float, sep=" ")[0]
##    Points[i-2,1] = np.fromstring(data[i], dtype=float, sep=" ")[1]
##    Points[i-2,2] = np.fromstring(data[i], dtype=float, sep=" ")[2]
    
##for i in range((PointIndex + 2), (PointIndex+TriangleIndex+2)):
##    Triangles[(i-2-PointIndex),0] = np.fromstring(data[i], dtype=int, sep=" ")[1]
##    Triangles[(i-2-PointIndex),1] = np.fromstring(data[i], dtype=int, sep=" ")[2]
##    Triangles[(i-2-PointIndex),2] = np.fromstring(data[i], dtype=int, sep=" ")[3]
    

def boundriesPython(Triangles,TriangleIndex):
    boundary_lines = np.zeros([TriangleIndex,2])
    index = 0
    for y in range (0,TriangleIndex):
        a = Triangles[y,0]
        b = Triangles[y,1]
        c = Triangles[y,2]
        
        if (not lt.repeated_lines_ab_01(Triangles,TriangleIndex,a,b,y)):
            if (not lt.repeated_lines_ab_12(Triangles,TriangleIndex,a,b,y)):
                if (not lt.repeated_lines_ab_20(Triangles,TriangleIndex,a,b,y)):
                    #boundary_lines[index,:] = [a,b]
                    boundary_lines[index,0] = a
                    boundary_lines[index,1] = b
                    index = index + 1

  
        if (not lt.repeated_lines_bc_01(Triangles,TriangleIndex,b,c,y)):
            if (not lt.repeated_lines_bc_12(Triangles,TriangleIndex,b,c,y)):
                if (not lt.repeated_lines_bc_20(Triangles,TriangleIndex,b,c,y)):
                    #boundary_lines[index,:] = [b,c]
                    boundary_lines[index,0] = b
                    boundary_lines[index,1] = c
                    index = index + 1

        if (not lt.repeated_lines_ca_01(Triangles,TriangleIndex,c,a,y)):
            if (not lt.repeated_lines_ca_12(Triangles,TriangleIndex,c,a,y)):
                if (not lt.repeated_lines_ca_20(Triangles,TriangleIndex,c,a,y)):
        #            #boundary_lines[index,:] = [c,a]
                    boundary_lines[index,0] = c
                    boundary_lines[index,1] = a
                    index = index + 1
        #print index
    return boundary_lines


def boundriesC(Triangles,TriangleIndex):
    boundary_linesC = np.zeros([TriangleIndex,2])
    index = 0
    ArrayCoppy = Triangles.flatten()
    abckz = np.array([0,0,0,0,0],"d")
    #print "triangles C 0 - 6 = ",ArrayCoppy[0:6]
    for y in range (0,TriangleIndex):
        #print "itterating Y in main python file"
        abckz[0] = Triangles[y,0]
        abckz[1] = Triangles[y,1]
        abckz[2] = Triangles[y,2]
        abckz[3] = y
        abckz[4] = TriangleIndex
        #print "abckz array = ", abckz        
        
        if (not (tl1.repeated_lines_ab_01(ArrayCoppy, abckz))):
            if (not (tl2.repeated_lines_ab_12(ArrayCoppy, abckz))):
                #print "number 0.2"
                if (not (tl3.repeated_lines_ab_20(ArrayCoppy, abckz))):
                    #print "number 1"
                    #boundary_lines[index,:] = [a,b]
                    boundary_linesC[index,0] = abckz[0]
                    boundary_linesC[index,1] = abckz[1]
                    index = index + 1
        if (not (tl4.repeated_lines_bc_01(ArrayCoppy, abckz))):
            if (not (tl5.repeated_lines_bc_12(ArrayCoppy, abckz))):
                #print "number 1.2"
                if (not (tl6.repeated_lines_bc_20(ArrayCoppy, abckz))):
                    #print "number 2"
                    #boundary_lines[index,:] = [b,c]
                    boundary_linesC[index,0] = abckz[1]
                    boundary_linesC[index,1] = abckz[2]
                    index = index + 1
        if (not (tl7.repeated_lines_ca_01(ArrayCoppy, abckz))):
            if (not (tl8.repeated_lines_ca_12(ArrayCoppy, abckz))):
        #        #print "numer 2.2"
                if (not (tl9.repeated_lines_ca_20(ArrayCoppy, abckz))):
                    #print "number 3"
                    #boundary_lines[index,:] = [c,a]
                    boundary_linesC[index,0] = abckz[2]
                    boundary_linesC[index,1] = abckz[0]
                    index = index + 1
        #print index
    boundary_linesC = boundary_linesC[~np.all(boundary_linesC == 0, axis=1)]
    return boundary_linesC

def IsolatePoints(EdgeMultiples,Points):
    print "in isolate points function"
    index = EdgeMultiples.size # this is a 2 by whatever array, this just saves a bit of time but returns overall size, not column size
    IsoPoints = np.zeros([index,3])
    IsoLines = np.zeros([index/2,6])
    IsoVectors = np.zeros([index/2,3])
    IsoVectorsHalf = np.zeros([index/2,3])
    IsoMidCords = np.zeros([index/2,3])
    #This for loop reshapes the array to give points and then coordinates
    for i in range(0,index/2): #goes to twice that
        ID = EdgeMultiples[i,0]
        IsoPoints[i,0] = Points[ID,0]
        IsoPoints[i,1] = Points[ID,1]
        IsoPoints[i,2] = Points[ID,2]
    for i in range(0,index/2):
        ID = EdgeMultiples[i,1]
        IsoPoints[i+index/2,0] = Points[ID,0]
        IsoPoints[i+index/2,1] = Points[ID,1]
        IsoPoints[i+index/2,2] = Points[ID,2]        
 
    for i in range(0,index/2): #goes to twice that
        IDA = EdgeMultiples[i,0]
        IDB = EdgeMultiples[i,1]
        IsoLines[i,0] = Points[IDA,0]
        IsoLines[i,1] = Points[IDA,1]
        IsoLines[i,2] = Points[IDA,2]
        IsoLines[i,3] = Points[IDB,0]
        IsoLines[i,4] = Points[IDB,1]
        IsoLines[i,5] = Points[IDB,2]

    for i in range (0,index/2):
        IsoVectors[i,0] = IsoLines[i,3] - IsoLines[i,0]
        IsoVectorsHalf[i,0] = IsoLines[i,0]/2
        IsoMidCords[i,0] = IsoLines[i,0] + IsoVectorsHalf[i,0] 
        IsoVectors[i,1] = IsoLines[i,4] - IsoLines[i,1]
        IsoVectorsHalf[i,1] = IsoLines[i,1]/2
        IsoMidCords[i,1] = IsoLines[i,1] + IsoVectorsHalf[i,1]
        IsoVectors[i,2] = IsoLines[i,5] - IsoLines[i,2]
        IsoVectorsHalf[i,2] = IsoLines[i,2]/2
        IsoMidCords[i,2] = IsoLines[i,2] + IsoVectorsHalf[i,2]
        
    return IsoPoints,IsoLines,IsoVectors,IsoVectorsHalf,IsoMidCords

#Triangles = np.array([[1,2,3],[4,5,6],[2,3,4],[3,2,1],[5,4,6]])   
#TriangleIndex = 5 

#start = time()
#EdgeMultiplesPy = boundriesPython(Triangles,TriangleIndex)
#stop = time()
#PythonTime = stop - start

##start = time()
##EdgeMultiplesC = boundriesC(Triangles,TriangleIndex)
##stop = time()
##CTime = stop - start
##print "C processing time = ", CTime

#print "python processing time = ", PythonTime

#if(np.array_equal(EdgeMultiplesC,EdgeMultiplesPy)):
#    print "arrays are equal"
#else:
#    print "not equal :/"

#np.savetxt("Carray.csv",EdgeMultiplesC,delimiter = ',' )
#np.savetxt("Pyarray.csv",EdgeMultiplesPy,delimiter = ',' )
#    
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1, projection='3d')
#
#ax.plot_trisurf(Points[:,0], Points[:,1], Points[:,2], triangles=Triangles, cmap=plt.cm.Spectral)
#
#plt.show()
