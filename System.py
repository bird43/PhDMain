import os
import DelStrip
import Para
import Void
import numpy as np
from time import time
import DiskCheck as DC
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg') # speeds up render
from scipy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import math
from scipy.cluster.vq import vq, kmeans, whiten
import stltool as TriRay

def PlotStuffMain(IP,CP,checked, PointA, ClusteredInternalVoids, SphereSurface, NoneObstructedVector, BestPoint, PreviousObservationPose, Points):
    DensitySQ = 10
    Density = 100
    y = ClusteredInternalVoids.size/3
    index = CP.size/3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ClusteredInternalVoids[:,0], ClusteredInternalVoids[:,1], ClusteredInternalVoids[:,2],color = 'g', s = 100)
    ax.scatter(Points[:,0], Points[:,1], Points[:,2],color = 'b', s = 10)
    #ax.plot_trisurf(PointA[:,0], PointA[:,1], PointA[:,2], triangles=Triangles, cmap=plt.cm.Spectral)
    for i in range(0,index):
        print checked[i]
        if checked[i] == 1: #if its greater than the specified radius!
            ax.scatter(PointA[i,0], PointA[i,1], PointA[i,2],color = 'b', s = 100) # external
        else:
            ax.scatter(PointA[i,0], PointA[i,1], PointA[i,2],color = 'r', s = 100) # internal
            
    for i in range (0,y):
        ax.plot([BestPoint[i,0],ClusteredInternalVoids[i,0]], [BestPoint[i,1],ClusteredInternalVoids[i,1]],zs=[BestPoint[i,2],ClusteredInternalVoids[i,2]])
        ax.scatter(BestPoint[i,0], BestPoint[i,1], BestPoint[i,2],color = 'r', s = 100)  
        ax.plot([BestPoint[i,0],PreviousObservationPose[0]], [BestPoint[i,1],PreviousObservationPose[1]],zs=[BestPoint[i,2],PreviousObservationPose[2]])
        for u in range (0,DensitySQ):#2pi * 10
            for v in range (0,DensitySQ):#pi * 10 make sure you -15
                if NoneObstructedVector[i*u*v+u+v] == 1:
                    #ax.plot([ClusteredInternalVoids[i,0], SphereSurface[((i*Density)+(u*DensitySQ+v)),0]], [ClusteredInternalVoids[i,1],SphereSurface[((i*Density)+(u*DensitySQ+v)),1]],zs=[ClusteredInternalVoids[i,2],SphereSurface[((i*Density)+(u*DensitySQ+v)),2]])  
                    i = i 
    plt.show()                
    return

def RunMeshLabRemotely(name):
	connection = Para.ssh("192.168.56.101", "ben", "starscape")
	connection.sendCommand(("cd ~/Dropbox/Robotics/ABB/DriveSoftware/System && xvfb-run meshlabserver -i " +name+".xyz -o "+name+".off -s Del.mlx"))


def main():
	#first need to determine if program is running in test mode and what hardware is being used
	print "mod5"
	print "Enter setup configuration details, hit enter for each one - "
	sit = raw_input("None active sim, Archived stitched data  - 'T' / None active with multiple PC's - 'M' / None active with pre computed BoundaryLines 'B' / Active test data, using arm and LiDAR - 'A'")
	hardware = raw_input("Running on RPi - 'R' / Running on VM Ubuntu - 'U'")
	MaxLength = int(raw_input("Enter max polygon edge length (150?)"))
	PlotEnable = raw_input("Display plots? Y1/Y2/N ")

	if (sit == "T" or sit == "B"):
		#name = "TotalCloudKM090"
		name = "SkullVoid"
		RunMeshLabRemotely(name)
		Points, Triangles = DelStrip.LoadData(name)

	NewTriangles = DelStrip.DeleteEdges(MaxLength,Points,Triangles) # Remove the convex hull stuff from the triangulation
	DelStrip.SaveMeshOff(Points,NewTriangles,name)
	TriangleIndex = NewTriangles.size / 3 #this may be the source of some erros
	print "Entering sit phase"

	if (sit == "T"):
		index = 10
		print "in T sit error!"
		BoundaryLines = Void.boundriesC(NewTriangles,TriangleIndex)
		np.savetxt('BoundaryLines.csv', BoundaryLines, delimiter=',') 
		IsoPoints,IsoLines,IsoVectors,IsoVectorsHalf,IsoMidCords = Void.IsolatePoints(BoundaryLines,Points)
		np.savetxt("IsoPoints.csv", IsoPoints, delimiter=",")
		np.savetxt("IsoLines.csv", IsoLines, delimiter=",")
		np.savetxt("IsoVectors.csv", IsoVectors, delimiter=",")
		np.savetxt("IsoVectorsHalf.csv", IsoVectorsHalf, delimiter=",")
		np.savetxt("IsoMidCords.csv", IsoMidCords, delimiter=",")
		Index = IsoLines.size/6
		PointA = np.zeros([Index,3])
		PointB = np.zeros([Index,3])
		PointA[:,0:2] = IsoLines[:,0:2]
		PointB[:,0:2] = IsoLines[:,3:5] 
		R = 9.1
		PreviousObservationPose = np.array([0,0,0,0,0,0])
		PlanePointA, PlanePointB, PlanePointC = DC.GenDiskPoints(IsoMidCords, R, IsoVectors)
		PlaneEquation = DC.CalculatePlanes(PlanePointA, PlanePointB, PlanePointC)
		IntersectionPoint = DC.CalculatePlaneIntersection(PlaneEquation, PointA, PointB)
		Checked, Delta = DC.CheckVectoABThroughDisc(IntersectionPoint, IsoMidCords, R, PointA)
		ClusteredInternalVoids = DC.GroupVoidsKMeans(PointA, 9)
		SphereSurface, NoneObstructedRays, BestPoint  = DC.GenerateScanVectors(ClusteredInternalVoids, Triangles, Points, PreviousObservationPose)
		np.savetxt("IntersectionPoint.csv", IntersectionPoint, delimiter=',')
		np.savetxt("IsoMidCords.csv", IsoMidCords, delimiter=',')
		np.savetxt("Checked.csv", Checked, delimiter=',')
		np.savetxt("PointA.csv", PointA, delimiter=',')
		np.savetxt("ClusteredInternalVoids.csv", ClusteredInternalVoids, delimiter=',')
		np.savetxt("SphereSurface.csv", SphereSurface, delimiter=',')
		np.savetxt("NoneObstructedRays.csv", NoneObstructedRays, delimiter=',')
		np.savetxt("BestPoint.csv", BestPoint, delimiter=',')
		np.savetxt("PreviousObservationPose.csv", PreviousObservationPose, delimiter=',')

	if(sit == "B"):	
		index = 10
		print "Loading Data............"
		BoundaryLines = np.genfromtxt("BoundaryLines.csv", delimiter=',')
		IsoPoints = np.genfromtxt("IsoPoints.csv", delimiter=',')
		IsoLines = np.genfromtxt("IsoLines.csv", delimiter=',')
		IsoVectors = np.genfromtxt("IsoVectors.csv", delimiter=',')
		IsoVectorsHalf = np.genfromtxt("IsoVectorsHalf.csv", delimiter=',')
		IsoMidCords = np.genfromtxt("IsoMidCords.csv", delimiter=',')

		IntersectionPoint = np.genfromtxt("IntersectionPoint.csv", delimiter=',')
		IsoMidCords = np.genfromtxt("IsoMidCords.csv", delimiter=',')
		Checked = np.genfromtxt("Checked.csv", delimiter=',')
		PointA = np.genfromtxt("PointA.csv", delimiter=',')
		ClusteredInternalVoids = np.genfromtxt("ClusteredInternalVoids.csv",delimiter=',' )
		SphereSurface = np.genfromtxt("SphereSurface.csv",delimiter=',' )
		NoneObstructedRays = np.genfromtxt("NoneObstructedRays.csv", delimiter=',')
		BestPoint = np.genfromtxt("BestPoint.csv", delimiter=',')
		PreviousObservationPose = np.genfromtxt("PreviousObservationPose.csv",delimiter=',' )

	if (PlotEnable == "Y2"):
		print "IntersectionPoint", IntersectionPoint
		print "IsoMidCords", IsoMidCords
		print "checked", Checked
		print "PointA", PointA
		print "ClusteredInternalVoids", ClusteredInternalVoids
		print "SphereSurface", SphereSurface
		print "NoneObstructedRays", NoneObstructedRays
		print "BestPoint", BestPoint
		print "PreviousObservationPose", PreviousObservationPose
		PlotStuffMain(IntersectionPoint, IsoMidCords ,Checked, PointA, ClusteredInternalVoids, SphereSurface, NoneObstructedRays, BestPoint, PreviousObservationPose, Points)



if __name__ == "__main__":
    main()