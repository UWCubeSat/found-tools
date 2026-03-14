import numpy as np
import random
import math
import sys

#CONVENTION: UNITS ARE IN KM AND RADIANS
#CONVENTION: GLOBAL COORDS ARE RIGHT-HANDED Z-UP; LOCAL COORDS ARE LEFT-HANDED Z-FORWARD

OUTPUT_FILE_PATH = "./pointOutput.txt"

#PARAMETERS
#   camera
xRes = 700
yRes = 600
sensorWidth = 0.036
focalLength = 0.05
#   point generation
angleRedundancy = 0.1 # ~5.7 degrees
pointNoise = 0
numPoints = 23 # number of points to generate

#CONSTANTS
pi = math.pi
Identity3 = np.array([[1,0,0],[0,1,0],[0,0,1]])
# 160 - 2000km above earth
LEOmin = 0.160+6.400
LEOmax = 2.000+6.400

#Ellipsoid defining matrix in world coords
#6,378,136.6 - Equatorial axes
#6,356,751.9 - Polar axis
Ap = np.array([[1/6.3781366**2, 0 , 0], [0, 1/6.3781366**2, 0], [0,0,1/6.3567519**2]])

# distribution makes x way more likely to have high values
# refactor to randomize which coord gets chosen first?
def generatePos(dist):
    x = random.random()*2 - 1
    y = (random.random()*2 - 1) * math.sqrt(1-x**2) # i think..?
    z = math.sqrt(1-x**2-y**2) # uhhh
    vec = np.array([x,y,z]) * dist
    return vec

# find analytical solution instead of brute forcing whether the horizon is visible maybe
def generateRot():
    x = random.random() * pi - pi/2
    y = random.random() * 2 * pi
    z = random.random() * 2 * pi
    rot = np.array([x,y,z])
    return rot

def seesHorizon(radius, distVecC, fov, redundancy):
    forward = np.array([0,0,1])
    #print(distVecC/np.linalg.norm(distVecC))
    #print(fov)
    angle = math.acos(forward.dot(distVecC/np.linalg.norm(distVecC)))
    #print(angle)
    sphereAngle = math.atan(radius/np.linalg.norm(distVecC))
    #print(sphereAngle)
    # print("\n")
    # print(f"angle: {angle}")
    # print(f"sphereAngle: {sphereAngle}")
    # print(f"fov: {fov}")
    # print(f"upper bound: {fov/2+sphereAngle}")
    # print(f"lower bound: {sphereAngle-fov/2}")
    # print((angle+redundancy) < (fov/2+sphereAngle)) and ((angle - redundancy) > (sphereAngle-fov/2))
    return ((angle+redundancy) < (fov/2+sphereAngle)) and ((angle - redundancy) > (sphereAngle-fov/2))

# this will not be the same matrix as the one CR generates, there's one (two?) degree(s?) of freedom
# ZXZ rotation
def generateTPC(rotation):
    #rotation = rotation * np.array([-1,-1,1])
    Xrot = np.array(
        [[1, 0, 0],
        [0, math.cos(rotation[0]), -math.sin(rotation[0])],
        [0, math.sin(rotation[0]), math.cos(rotation[0])]])
    adjust = np.array(
        [[1, 0, 0],
        [0, math.cos(pi/2), -math.sin(pi/2)],
        [0, math.sin(pi/2), math.cos(pi/2)]])
    # we use a z rotation cause we're doing ZXZ
    Yrot = np.array(
        [[math.cos(rotation[1]), 0, math.sin(rotation[1])],
        [0, 1, 0],
        [-math.sin(rotation[1]), 0, math.cos(rotation[1])]])
    #     [[math.cos(rotation[1]), 0, math.sin(rotation[1])],
    #     [0, 1, 0],
    #     [-math.sin(rotation[1]), 0, math.cos(rotation[1])]])
    Zrot = np.array(
        [[math.cos(rotation[2]), -math.sin(rotation[2]), 0],
        [math.sin(rotation[2]), math.cos(rotation[2]), 0],
        [0, 0, 1]])

    invertZ = np.array([[1,0,0],[0,1,0],[0,0,-1]]) # DON'T FORGET TO CHANGE BACK

    # RIGHT HAND GLOBAL COORDS (Z UP) -> LEFT HAND LOCAL CAM COORDS (Z FORWARD)
    TPC = Zrot.dot(Xrot.dot(Yrot.dot(adjust.dot(invertZ))))
    return np.transpose(TPC)

#generates the conic in image coords
def generateConic(rc, Ap, TPC):
    Ac = np.transpose(TPC).dot(Ap.dot(TPC))
    C = Ac.dot(np.outer(rc, rc).dot(Ac)) - (rc.dot(Ac.dot(rc))*Identity3 - Identity3).dot(Ac) 
    return C

def generateInvCameraMat(sensorWidth, xRes):
    pixelSize = sensorWidth/xRes

    dx = focalLength/pixelSize
    dy = dx # square pixels

    KInv = np.array([[1/dx, 0,      -(xRes/2)/(dx) ],
                    [0,    1/dy,   -(yRes/2)/(dy) ],
                    [0,    0,      1           ]])
    return KInv

#generates the conic in pixel coords
def generateCalibratedConic(C, KInv):
    calibratedC = np.transpose(KInv).dot(C.dot(KInv))
    calibratedC = calibratedC/calibratedC[0][0]
    # print("\n\n")
    # print("Calibrated conic:")
    # # plug this into desmos to see the curve in pixel coords
    # print(f"{calibratedC[0][0]}x^2 + {calibratedC[0][1]}*2xy + {calibratedC[1][1]}y^2 + {calibratedC[0][2]}*2x + {calibratedC[1][2]}*2y+{calibratedC[2][2]} = 0") 
    # print("\n\n")
    return calibratedC

def noise(pointNoise, points):
    return points # trust me it works

def generatePoints(calibratedConic, pointNoise, numPoints):
    points = np.zeros((numPoints, 2))
    for i in range(numPoints):
        x = random.random() * xRes
        a = calibratedConic[1][1]
        b = calibratedConic[0][1]*2*x+2*calibratedConic[1][2]
        c = (calibratedConic[0][0]*x*x+calibratedConic[0][2]*2*x+calibratedConic[2][2])
        plusorminus = round(random.random())*2-1
        det = b**2 - 4*a*c
        y = 0
        if(det>0):
                y = (-b + plusorminus*math.sqrt(det))/(2*a)
        counter = 0
        while (True): # make sure there are real roots
            x = random.random() * xRes
            counter += 1
            if (counter > numPoints*1000):
                points[i] = None
                return points
            a = calibratedConic[1][1]
            b = calibratedConic[0][1]*2*x+2*calibratedConic[1][2]
            c = (calibratedConic[0][0]*x*x+calibratedConic[0][2]*2*x+calibratedConic[2][2])
            det = (b)**2 - 4*a*c
            if (det < 0):
                continue
            y = (-b + plusorminus*math.sqrt(det))/(2*a)
            if (y<0 or y>yRes):
                plusorminus = -plusorminus
                y = (-b + plusorminus*math.sqrt(det))/(2*a)
                if (y<0 or y>yRes):
                    continue
            break
        points[i] = np.array([x,y])
        
        
        #print(f"{{static_cast<decimal>({x}), static_cast<decimal>({y})}},")
    points = noise(pointNoise, points)
    return points

# position in world coords, local rotation
def posrotmain(positionx, positiony, positionz, rotationx, rotationy, rotationz, numPoints, pointNoise, angleRedundancy, xRes, yRes, sensorWidth, focalLength):
    rp = np.array([positionx, positiony, positionz])
    rotation = np.array([rotationx, rotationy, rotationz])
    print("\n\n")
    print(f"rp: {rp}")
    print(f"rotation: {rotation*180/(math.pi)}")
    TPC = generateTPC(rotation) 
    rc = TPC.dot(rp)
    print(f"rc: {rc}")
    print("\n\n")
    TCP = np.transpose(TPC)
    positions = np.array([[0.097372,-0.315722,0.943843],[-0.15943,0.931154,0.327925],[-0.982396,-0.182407,0.040333]])
    print(f"positions:\n{positions}")
    print(f"calc TCP:\n{TCP}")
    print(f"det: {np.linalg.det(TCP)}")
    print("\n\n")
    fov = 2*math.atan(sensorWidth/(2*focalLength))
    # if not seesHorizon(math.sqrt(1/Ap[0][0]), -1*rc, fov, angleRedundancy):
    #     raise ValueError("Camera can't see the horizon idiot!")
    C = generateConic(rc, Ap, TPC)
    KInv = generateInvCameraMat(sensorWidth, xRes)
    calibratedC = generateCalibratedConic(C, KInv)
    points = generatePoints(calibratedC, pointNoise, numPoints)
    if (not points.all()):
        print("AAAAARGHHHH")
        return False
    # appends
    with open(OUTPUT_FILE_PATH, "a") as f:
        rotation = rotation*180/(math.pi)
        rc = rc*100000
        rp = rp*100000
        f.write(f"\n\nPOINTS FOR\nlocal (rc) [{rc[0]}, {rc[1]}, {rc[2]}]\nglobal (rp) [{rp[0]}, {rp[1]}, {rp[2]}] \nwith rotation [{rotation[0]}, {rotation[1]}, {rotation[2]}]:\n")
        for point in points:
            f.write(f"{{static_cast<decimal>({point[0]}), static_cast<decimal>({point[1]})}},")
        f.write("\nTPC:\n{")
        for row in TPC:
            f.write(f"{row[0]}, {row[1]}, {row[2]},\n")
        f.write("}")
        f.write("\n\nCalibrated conic equation: ")
        f.write(f"{calibratedC[0][0]}x^2 + {calibratedC[0][1]}*2xy + {calibratedC[1][1]}y^2 + {calibratedC[0][2]}*2x + {calibratedC[1][2]}*2y+{calibratedC[2][2]} = 0")
        f.write(f"\n\nOther settings:\nnum points: {numPoints}\npoint noise: {pointNoise}\nangle redundancy: {angleRedundancy}\nresolution: {xRes}x{yRes}\nsensor width: {sensorWidth}\nfocal length: {focalLength}\nFOV: {fov}\nAOR: {TPC.dot([0,0,1])}\n\n------------------------------------------------")
    return True
# position in world coords, local rotation
def posmain(positionx, positiony, positionz, numPoints, pointNoise, angleRedundancy, xRes, yRes, sensorWidth, focalLength):
    rp = np.array([positionx, positiony, positionz])
    rotation = generateRot()
    TPC = generateTPC(rotation)
    rc = TPC.dot(rp)
    fov = 2*math.atan(sensorWidth/(2*focalLength))
    while not seesHorizon(math.sqrt(1/Ap[0][0]), -1*rc, fov, angleRedundancy):
        #print(rotation)
        rotation = generateRot()
        TPC = np.linalg.inv(generateTPC(rotation))
        rc = TPC.dot(rp)
    while (not posrotmain(positionx, positiony, positionz, rotation[0], rotation[1], rotation[2], numPoints, pointNoise, angleRedundancy, xRes, yRes, sensorWidth, focalLength)): # this is clean code. 
        rotation = generateRot()
        TPC = generateTPC(rotation)
        rc = TPC.dot(rp)
        fov = 2*math.atan(sensorWidth/(2*focalLength))
        while not seesHorizon(math.sqrt(1/Ap[0][0]), -1*rc, fov, angleRedundancy):
            #print(rotation)
            rotation = generateRot()
            TPC = np.linalg.inv(generateTPC(rotation))
            rc = TPC.dot(rp)

# position in world coords, local rotation
def rawmain(numPoints, pointNoise, angleRedundancy, xRes, yRes, sensorWidth, focalLength):
    dist = LEOmin + random.random()*(LEOmax-LEOmin)
    position = generatePos(dist)
    posmain(position[0], position[1], position[2], numPoints, pointNoise, angleRedundancy, xRes, yRes, sensorWidth, focalLength)


if len(sys.argv) == 14:
    posrotmain(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])/180*pi, float(sys.argv[5])/180*pi, float(sys.argv[6])/180*pi, int(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9]), float(sys.argv[10]), float(sys.argv[11]), float(sys.argv[12]), float(sys.argv[13]))
elif len(sys.argv) == 11:
    posmain(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9]), float(sys.argv[10]))
elif len(sys.argv) == 8:
    rawmain(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]))
else:
    print("Usage: point-generator.py positionx, positiony, positionz, rotationx, rotationy, rotationz, numPoints, pointNoise, angleRedundancy, xRes, yRes, sensorWidth, focalLength")
    print("Position and rotation optional")


