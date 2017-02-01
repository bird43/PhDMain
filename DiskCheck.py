import numpy as np 
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg') # speeds up render
from scipy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import math
from scipy.cluster.vq import vq, kmeans, whiten
import stltool as TriRay


def GenDiskPoints(CentrePoint, R, LineVector):
    #Generates points A, B and C on a disk defined by the centre point, Radius and a line vector
    # which is the vector that is perpendicular to the plane of the disc, and goes through the centre
    index = LineVector.size/3
    U = np.zeros([index,3])
    Perp = np.zeros([index,3])
    V = np.zeros([index,3])
    CrossP = np.zeros([index,3])
    PlanePointA = np.zeros([index,3])
    PlanePointB = np.zeros([index,3])
    PlanePointC = np.zeros([index,3])
    for i in range (0,index/2):
        magni = ((LineVector[i,0]**2)+(LineVector[i,1]**2)+(LineVector[i,2]**2))**0.5
        V[i,:] = LineVector[i,:]/magni
        Perp[i,0] = 2
        Perp[i,1] = 2
        Perp[i,2] = (-(2*LineVector[i,0])-(2*LineVector[i,1]))/LineVector[i,2]
        magni = ((Perp[i,0]**2)+(Perp[i,1]**2)+(Perp[i,2]**2))**0.5
        U[i,:] = Perp[i,:]/magni
        CrossP[i,0] = ((U[i,1]*V[i,2])-(U[i,2]*V[i,1]))
        CrossP[i,1] = ((U[i,2]*V[i,0])-(U[i,0]*V[i,2]))
        CrossP[i,2] = ((U[i,0]*V[i,1])-(U[i,1]*V[i,0]))
        PlanePointA[i,:] = R*np.cos(1)*U[i,:] + R*np.sin(1)*CrossP[i,:] + CentrePoint[i,:]
        PlanePointC[i,:] = R*np.cos(2)*U[i,:] + R*np.sin(2)*CrossP[i,:] + CentrePoint[i,:]
        PlanePointB[i,:] = CentrePoint[i,:]
	return PlanePointA, PlanePointB, PlanePointC


def CalculatePlanes(PlanePointA, PlanePointB, PlanePointC):
    #Calculates the plane defined by the points A,B and C which lie on the disc
    index = PlanePointA.size/3
#GenVectors
    VectorA = np.zeros([index,3])
    VectorB = np.zeros([index,3])
    CrossP = np.zeros([index,3])
    Plane = np.zeros([index,4])
    for i in range (0, index):
        VectorA[i,0] = PlanePointA[i,0] - PlanePointB[i,0]
        VectorA[i,1] = PlanePointA[i,1] - PlanePointB[i,1]
        VectorA[i,2] = PlanePointA[i,2] - PlanePointB[i,2]
        VectorB[i,0] = PlanePointC[i,0] - PlanePointB[i,0]
        VectorB[i,1] = PlanePointC[i,1] - PlanePointB[i,1]
        VectorB[i,2] = PlanePointC[i,2] - PlanePointB[i,2]	
#cross product of the two vecotrs
        CrossP[i,0] = ((VectorA[i,1]*VectorB[i,2])-(VectorA[i,2]*VectorB[i,1]))
        CrossP[i,1] = ((VectorA[i,2]*VectorB[i,0])-(VectorA[i,0]*VectorB[i,2]))
        CrossP[i,2] = ((VectorA[i,0]*VectorB[i,1])-(VectorA[i,1]*VectorB[i,0]))
        Plane[i,0] = CrossP[i,0]
        Plane[i,1] = CrossP[i,1]
        Plane[i,2] = CrossP[i,2]
        Plane[i,3] = CrossP[i,0]*PlanePointB[i,0] + CrossP[i,1]*PlanePointB[i,1] + CrossP[i,2]*PlanePointB[i,2]
    return Plane

def CalculatePlaneIntersection(PlaneEquation, PointA, PointB):
    #Calculates the point at which the vector PointB-PointA intersects the plane defined by the plane equation
    index = PointA.size/3
    Vectors = np.zeros([index,3])
    Tee = np.zeros(index)
    NoTee = np.zeros(index)
    IntersectionScalar = np.zeros(index)
    IntersectionPoint = np.zeros([(index*index),3])
    for i in range (0, index):
        Vectors[i,0] = PointB[i,0] - PointA[i,0]
        Vectors[i,1] = PointB[i,1] - PointA[i,1]
        Vectors[i,2] = PointB[i,2] - PointA[i,2]
        Tee[i] = Vectors[i,0] + Vectors[i,1] + Vectors[i,2]
        NoTee[i] = PointA[i,0] + PointA[i,1] + PointA[i,2]
    for i in range (0, index):
        for j in range (0, index):
            print index, i
            IntersectionScalar[i] = (PlaneEquation[i,3] - NoTee[j]) / Tee[j]  
            IntersectionPoint[(i*index+j),0] = PointA[j,0] + (IntersectionScalar[j] * Vectors[j,0])
            IntersectionPoint[(i*index+j),1] = PointA[j,1] + (IntersectionScalar[j] * Vectors[j,1])
            IntersectionPoint[(i*index+j),2] = PointA[j,2] + (IntersectionScalar[j] * Vectors[j,2])
    return IntersectionPoint

def CheckVectoABThroughDisc(IP, CP, R, PointA):
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    index = CP.size/3
    Delta = np.zeros(index**2)
    checked = np.zeros(index)
    #ax.plot_trisurf(AllPoints[:,0], AllPoints[:,1], AllPoints[:,2], triangles=Triangles, cmap=plt.cm.Spectral)
    for i in range(0,index):
        for j in range(0,index):
            Delta[(i*index+j)] = (((IP[j,0]-CP[i,0])**2)+((IP[j,1]-CP[i,1])**2)+((IP[j,2]-CP[i,2])**2))**0.5
            if Delta[(i*index+j)] < R:
                checked[i] = 1
                #ax.scatter(PointA[i,0], PointA[i,1], PointA[i,2],color = 'b', s = 10)
                #print CP[i,:]
            else:
                checked[i] = 0
                #ax.scatter(PointA[i,0], PointA[i,1], PointA[i,2],color = 'r', s = 10)

    #plt.show()
    #print "DeltaMean = ", np.nanmean(Delta)
    #print "Trimmed Delta at 10% =", stats.trim_mean(Delta, 0.1) #5% either way
    #print "Trimmed Delta at 5% =", stats.trim_mean(Delta, 0.05)
    #print "Trimmed Delta at 3% =", stats.trim_mean(Delta, 0.03)
    print "Trimmed Delta at 1% =", stats.trim_mean(Delta, 0.01)
    return checked, Delta

def PlotStuffBasic(IP,CP,checked, PointA, ClusteredInternalVoids, SphereSurface, NoneObstructedVector, BestPoint, PreviousObservationPose):
    DensitySQ = 10
    Density = 100
    y = ClusteredInternalVoids.size/3
    index = CP.size/3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ClusteredInternalVoids[:,0], ClusteredInternalVoids[:,1], ClusteredInternalVoids[:,2],color = 'g', s = 100)
    #ax.plot_trisurf(AllPoints[:,0], AllPoints[:,1], AllPoints[:,2], triangles=Triangles, cmap=plt.cm.Spectral)
    #for i in range(0,index):
        #print checked[i]
        #if checked[i] == 1: #if its greater than the specified radius!
            #ax.scatter(PointA[i,0], PointA[i,1], PointA[i,2],color = 'b', s = 100) # external
        #else:
            #ax.scatter(PointA[i,0], PointA[i,1], PointA[i,2],color = 'r', s = 100) # internal
            
    for i in range (0,y):
        ax.plot([BestPoint[i,0],ClusteredInternalVoids[i,0]], [BestPoint[i,1],ClusteredInternalVoids[i,1]],zs=[BestPoint[i,2],ClusteredInternalVoids[i,2]])
        ax.scatter(BestPoint[i,0], BestPoint[i,1], BestPoint[i,2],color = 'r', s = 100)  
        ax.plot([BestPoint[i,0],PreviousObservationPose[0]], [BestPoint[i,1],PreviousObservationPose[1]],zs=[BestPoint[i,2],PreviousObservationPose[2]])
        for u in range (0,DensitySQ):#2pi * 10
            for v in range (0,DensitySQ):#pi * 10 make sure you -15
                if NoneObstructedVector[i*u*v+u+v] == 1:
                    #ax.plot([ClusteredInternalVoids[i,0], SphereSurface[((i*Density)+(u*DensitySQ+v)),0]], [ClusteredInternalVoids[i,1],SphereSurface[((i*Density)+(u*DensitySQ+v)),1]],zs=[ClusteredInternalVoids[i,2],SphereSurface[((i*Density)+(u*DensitySQ+v)),2]])  
                    i = i 
    return

def PlotStuff(CentrePointOfDisc, R, PointA, PointB, Checked):
    p0 = np.zeros(3)
    p1 = np.zeros(3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    plt.show()
    index = 1
    #plots the disk, vector being checked that contains the centre point of the disk 
    # and the vector being checked which is made up of PointB - PointA
    #index = PointA.size/3
    for i in range(0, index):
        p0[:] = CentrePointOfDisc[i,:] - 0.1
        p1[:] = CentrePointOfDisc[i,:] + 0.1
        #vector in direction of axis
        v = p1 - p0
        #find magnitude of vector
        mag = norm(v)
        #unit vector in direction of axis
        v = v / mag
        #make some vector not in the same direction as v
        not_v = np.array([1, 0, 0])
        if (v == not_v).all():
            not_v = np.array([0, 1, 0])
        #make vector perpendicular to v
        n1 = np.cross(v, not_v)
        #normalize n1
        n1 /= norm(n1)
        #make unit vector perpendicular to v and n1
        n2 = np.cross(v, n1)
        #surface ranges over t from 0 to length of axis and 0 to 2*pi
        t = np.linspace(0, mag, 100)
        theta = np.linspace(0, 2 * np.pi, 100)
        #use meshgrid to make 2d arrays
        t, theta = np.meshgrid(t, theta)
        #generate coordinates for surface
        X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        #ax.plot_surface(X, Y, Z)
        
        #plot axis
        #ax.plot(*zip(p0, p1), color = 'red')
        if Checked[i] == 1:
            ax.plot(*zip(PointA,PointB), color = 'blue')

def LoadData():

    IsoLines = np.genfromtxt ('IsoLines.csv', delimiter=",") #CHANGE ME!!!!
    index = IsoLines.size/6
    PointA = np.zeros([index,3])
    PointB = np.zeros([index,3])
    for i in range (0,index):
        PointA[i,0] = IsoLines[i,0]
        PointA[i,1] = IsoLines[i,1]
        PointA[i,2] = IsoLines[i,2]
        PointB[i,0] = IsoLines[i,3]
        PointB[i,1] = IsoLines[i,4]
        PointB[i,2] = IsoLines[i,5]
    IsoVectors = np.genfromtxt ('IsoVectors.csv', delimiter=",")
    IsoMidCords = np.genfromtxt ('IsoMidCords.csv', delimiter=",")
    AllPoints = np.genfromtxt ('Points.csv', delimiter=",")
    Triangles = np.genfromtxt ('Triangles.csv', delimiter=",")
    return PointA, PointB, IsoVectors, IsoMidCords, AllPoints, Triangles, IsoLines
    
def GroupVoidsRegionGrowing(PointCloud, ClusterSize, Variation): #uses region growing algorithm
    y,x = PointCloud.shape
    SurfaceCluster = np.zeros((y,3))
    Counter = 0
    VecMagOld = math.sqrt(math.pow(PointCloud[0,0],2) + math.pow(PointCloud[0,1],2) + math.pow(PointCloud[0,2],2))
    #print VecMagOld    
    for i in range (1,y):
        Counter += 1
        VecMag = math.sqrt(math.pow(PointCloud[i,0],2) + math.pow(PointCloud[i,1],2) + math.pow(PointCloud[i,2],2))
        if VecMag > VecMagOld + (VecMagOld * Variation):
            try:
                SurfaceCluster[i,0] = np.mean(PointCloud[i:i+Counter,0])
                SurfaceCluster[i,1] = np.mean(PointCloud[i:i+Counter,1])
                SurfaceCluster[i,2] = np.mean(PointCloud[i:i+Counter,2])
                #print SurfaceCluster[i,:]
                Counter = 0
            except IndexError:
                break
        if VecMag > VecMagOld - (VecMagOld * Variation):
            try:
                SurfaceCluster[i,0] = np.mean(PointCloud[i:i+Counter,0])
                SurfaceCluster[i,1] = np.mean(PointCloud[i:i+Counter,1])
                SurfaceCluster[i,2] = np.mean(PointCloud[i:i+Counter,2])
                #print SurfaceCluster[i,:]
                Counter = 0
            except IndexError:
                break
        if Counter > math.pow(ClusterSize,2):
            try:
                SurfaceCluster[i,0] = np.mean(PointCloud[i:i+Counter,0])
                SurfaceCluster[i,1] = np.mean(PointCloud[i:i+Counter,1])
                SurfaceCluster[i,2] = np.mean(PointCloud[i:i+Counter,2])
                Counter = 0
            except IndexError:
                break
        VecMagOld = VecMag
    print SurfaceCluster
    SurfaceCluster[SurfaceCluster==0] = np.nan 
    SurfaceCluster = SurfaceCluster[~np.isnan(SurfaceCluster).any(axis=1)] 
    print SurfaceCluster
    return SurfaceCluster
    
def GroupVoidsKMeans(PointCloud,ClusterSize):
    Clustered, Distortion = kmeans(PointCloud,ClusterSize,iter=1)
    print Distortion
    return Clustered
    
def GenerateScanVectors(ClusteredInternalVoids, Triangles, AllPoints, PreviousObservationPose):
    #fig2 = plt.figure()
    #ax = fig2.add_subplot(111, projection='3d')
    DensitySQ = 10
    Density = 100
    r = 10
    Divisor = 6.28/DensitySQ
    print Divisor
    y = ClusteredInternalVoids.size / 3
    TriIndex = Triangles.size / 3
    index = ((y+1) * Density)
    SphereSurface = np.zeros((index,3))
    SphereVector = np.zeros((index,3))
    NoneObstructedRays = np.ones(index)
    BestPoint = np.zeros((y+1,3))
    Distance = np.zeros(3)

    #http://www.lighthouse3d.com/tutorials/maths/ray-triangle-intersection/
    #http://mymathforum.com/algebra/3066-what-parametric-equation-sphere.html
    #MollerTrumbore intersection algorithm    
    for i in range (0,y):
        for u in range (0,DensitySQ):#2pi * 10
            for v in range (0,DensitySQ):#pi * 10 make sure you -15
                U = u
                V = v
                SphereSurface[((i*Density)+(u*DensitySQ+v)),0] = ClusteredInternalVoids[i,0] + r * np.sin(U*Divisor) * np.cos(V*Divisor)
                SphereSurface[((i*Density)+(u*DensitySQ+v)),1] = ClusteredInternalVoids[i,1] + r * np.cos(U*Divisor) * np.cos(V*Divisor)
                SphereSurface[((i*Density)+(u*DensitySQ+v)),2] = ClusteredInternalVoids[i,2] + r * np.sin(V*Divisor)
                SphereVector[((i*Density)+(u*DensitySQ+v)),0] = SphereSurface[((i*Density)+(u*DensitySQ+v)),0] - ClusteredInternalVoids[i,0]
                SphereVector[((i*Density)+(u*DensitySQ+v)),1] = SphereSurface[((i*Density)+(u*DensitySQ+v)),1] - ClusteredInternalVoids[i,1]
                SphereVector[((i*Density)+(u*DensitySQ+v)),2] = SphereSurface[((i*Density)+(u*DensitySQ+v)),2] - ClusteredInternalVoids[i,2]
                #ax.plot([ClusteredInternalVoids[i,0], SphereSurface[((i*25)+(u*5+v)),0]], [ClusteredInternalVoids[i,1],SphereSurface[((i*25)+(u*5+v)),1]],zs=[ClusteredInternalVoids[i,2],SphereSurface[((i*25)+(u*5+v)),2]])
    for i in range (0,TriIndex):
        v1 = AllPoints[Triangles[i,0],:]
        v2 = AllPoints[Triangles[i,1],:]
        v3 = AllPoints[Triangles[i,2],:]
        for j in range (0, y):    
            for k in range (0,Density):
                if TriRay.ray_triangle_intersection(ClusteredInternalVoids[j,:], SphereVector[j*Density+k,:], (v1,v2,v3)) == True:
                    print "obstruction found, i =", i, "j =", j,"k =", k
                    NoneObstructedRays[j*Density+k] = 0
                    

    for i in range (0,y):#y = ClusteredInternalVoids.size / 3
        Distance[0] = SphereSurface[y,0] - PreviousObservationPose[0]
        Distance[1] = SphereSurface[y,1] - PreviousObservationPose[1]
        Distance[2] = SphereSurface[y,2] - PreviousObservationPose[2]
        DisMagOld = np.linalg.norm(Distance)
        BestPoint[i,:] = SphereSurface[y,:]
        for j in range (0,Density):#number of vectors per point
            if NoneObstructedRays[i*Density+j] == 1:
                Distance[0] = SphereSurface[i*Density+j,0] - PreviousObservationPose[0]
                Distance[1] = SphereSurface[i*Density+j,1] - PreviousObservationPose[1]
                Distance[2] = SphereSurface[i*Density+j,2] - PreviousObservationPose[2]
                DisMagNew = np.linalg.norm(Distance)
                if DisMagNew <= DisMagOld:
                    BestPoint[i,:] = SphereSurface[i*Density+j,:]
                    
    return SphereSurface, NoneObstructedRays, BestPoint
    
def main():    
    PointA, PointB, LineVector, CentrePoint, AllPoints, Triangles, IsoLines = LoadData()#centre point = isomidcords linevector = isovector
    R = 9.1
    PreviousObservationPose = np.array([0,0,0,0,0,0])

    PlanePointA, PlanePointB, PlanePointC = GenDiskPoints(CentrePoint, R, LineVector)

    PlaneEquation = CalculatePlanes(PlanePointA, PlanePointB, PlanePointC)

    IntersectionPoint = CalculatePlaneIntersection(PlaneEquation, PointA, PointB)

    Checked, Delta = CheckVectoABThroughDisc(IntersectionPoint, CentrePoint, R, PointA)

    ClusteredInternalVoids = GroupVoidsKMeans(PointA, 9)

    SphereSurface, NoneObstructedRays, BestPoint  = GenerateScanVectors(ClusteredInternalVoids, Triangles, AllPoints, PreviousObservationPose)

    PlotStuffBasic(IntersectionPoint, CentrePoint ,Checked, PointA, ClusteredInternalVoids, SphereSurface, NoneObstructedRays, BestPoint, PreviousObservationPose)


if __name__ == "__main__":
    main()

#ClusteredInternalVoids = GroupVoids(PointCloud, 9, 0.1)

#PlotStuff(CentrePoint, R, PointA, PointB, Checked)


#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1, projection='3d')
#ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
#ax.plot_trisurf(AllPoints[:,0], AllPoints[:,1], AllPoints[:,2], triangles=Triangles, cmap=plt.cm.Spectral)
#ax.scatter(PointA[:,0], PointA[:,1], PointA[:,2])
#ax.scatter(PointB[:,0], PointB[:,1], PointB[:,2])
#ax2.scatter(PointA[:,0], PointA[:,1], PointA[:,2], c = 'g', s = 1)
#ax2.scatter(PointB[:,0], PointB[:,1], PointB[:,2], c = 'r', s = 1)
#plt.show()
