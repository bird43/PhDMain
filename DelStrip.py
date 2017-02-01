#script to get rid of the nonsense in a delaunay triangulation, basically hwat meshlab does when you ask it. 
#use TetGen to compute actual delaunay triangulation, call from command line using os
import numpy as np

def LoadData(name):	
	#offSource = open("MeshWithGoodHoles.off", "r")
	string = name + ".off"
	offSource = open(string, "r")
	#data = offSource.readlines()
	data = offSource.read().split('\n')
	#Indexing = data[1]
	PointIndex = np.fromstring(data[1], dtype=int, sep=" ")[0]
	TriangleIndex = np.fromstring(data[1], dtype=int, sep=" ")[1]

	Points = np.zeros([PointIndex,3])
	Triangles = np.zeros([TriangleIndex,3])

	for i in range(2, PointIndex+2):
	    Points[i-2,0] = np.fromstring(data[i], dtype=float, sep=" ")[0]
	    Points[i-2,1] = np.fromstring(data[i], dtype=float, sep=" ")[1]
	    Points[i-2,2] = np.fromstring(data[i], dtype=float, sep=" ")[2]
    
	for i in range((PointIndex + 2), (PointIndex+TriangleIndex+2)):
	    Triangles[(i-2-PointIndex),0] = np.fromstring(data[i], dtype=int, sep=" ")[1]
	    Triangles[(i-2-PointIndex),1] = np.fromstring(data[i], dtype=int, sep=" ")[2]
	    Triangles[(i-2-PointIndex),2] = np.fromstring(data[i], dtype=int, sep=" ")[3]
	return Points, Triangles

def DeleteEdges(MaxLength,Points,Triangles):
	index = Triangles.size/3
	PointA = np.zeros(3)
	PointB = np.zeros(3)
	PointC = np.zeros(3)
	for i in range (0,index):
		PointA[:] = Points[Triangles[i,0],:]
		PointB[:] = Points[Triangles[i,1],:]
		PointC[:] = Points[Triangles[i,2],:]
		MagA = np.linalg.norm((PointA[:] - PointB[:]))
		MagB = np.linalg.norm((PointB[:] - PointC[:]))
		MagC = np.linalg.norm((PointC[:] - PointA[:]))
		
		if (MagA > MaxLength):
			Triangles[i,:] = [0.0,0.0,0.0]
		if (MagB > MaxLength):
			Triangles[i,:] = [0.0,0.0,0.0]
		if (MagC > MaxLength):
			Triangles[i,:] = [0.0,0.0,0.0]

	mask = np.all(np.isnan(Triangles), axis=1)
	Triangles[~mask]
	return Triangles

def SaveMeshOff(Points,NewTriangles,name):
	NewMesh = open((name + "New.off"), "w")
	IT = NewTriangles.size/3
	IP = Points.size/3
	NewMesh.write("OFF\n")
	NewMesh.write(str(IP) + " " + str(IT) + " 0\n")
	for i in range (0,IP):	
		NewMesh.write(str(Points[i,0]) + " " + str(Points[i,1]) + " " + str(Points[i,2]) + "\n")
	for i in range (0,IT):
		NewMesh.write("3 " + str(NewTriangles[i,0]) + " " + str(NewTriangles[i,1]) + " " + str(NewTriangles[i,2]) + "\n")
	NewMesh.close()
	return

def Node2XYZ(name):
	NodeSource = open((name + '.node') , "r")
	data = NodeSource.read().split('\n')
	index = np.fromstring(data[1], dtype=int, sep=" ")[0]
	Points = np.zeros([index,3])

	for i in range(2, index+2):
	    Points[i-2,0] = np.fromstring(data[i], dtype=float, sep=" ")[1]
	    Points[i-2,1] = np.fromstring(data[i], dtype=float, sep=" ")[2]
	    Points[i-2,2] = np.fromstring(data[i], dtype=float, sep=" ")[3]

	np.savetxt((name+".xyz"), Points, delimiter=" ")
	return

def XYZ2Node(name):
	#probably redundant....
	Points = np.genfromtxt((name +'.xyz'), delimiter=' ')
	SaveNode(name,Points)
	return Points

def SaveNode(name,Points):
	Node = open((name+".node"), "w")
	IP = Points.size/3
	Node.write("\n")
	Node.write("	" + str(IP) + " " + "3 0 0\n")
	for i in range (0,IP):	
		Node.write("		" + str(i) + " " + str(Points[i,0]) + " " + str(Points[i,1]) + " " + str(Points[i,2]) + "\n")
	return

def LoadXYZ(name):
	PC = np.genfromtxt((name +'.xyz'), delimiter=' ')
	return PC

def LoadFaces(name):
	Data = np.genfromtxt((name + '.1.face') , skip_header=1, skip_footer=1)
	#print Data
	#FaceSource = open((name + '.1.face') , "r")
	#data = FaceSource.read().split('\n')
	index = Data.size/5 
	print index
	Triangles = np.zeros([index,3])
	for i in range(0,index):
		Triangles[i,0] = Data[i,1]
		Triangles[i,1] = Data[i,2]
		Triangles[i,2] = Data[i,3]
	#for i in range(1, index):
	#	Triangles[i-1,0] = np.fromstring(data[i], dtype=int, sep="")[0]
	#	Triangles[i-1,1] = np.fromstring(data[i], dtype=int, sep="")[1]
	#	Triangles[i-1,2] = np.fromstring(data[i], dtype=int, sep="")[2]
	#	print Triangles[i,:]
	print Triangles
	return Triangles

def main():
	name = "TotalCloudKM090"
	Triangles = LoadFaces(name)
	#MaxLength = 200
	#name = "DelStripTester"
	#Points, Triangles = LoadData(name)
	#NewTriangles = DeleteEdges(MaxLength,Points,Triangles)
	#SaveMeshOff(Points,Triangles)

    

if __name__ == "__main__":
    main()