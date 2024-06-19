# Meshroom implementation by David Castano for Windows. This adaptation of his code by Dr. Anirudh Rao (Imperial College London) to run on Linux machines. 
# Updated by Abdurrahman Yilmaz (ayilmaz@lincoln.ac.uk) v03

import sys
import os , os.path
#import shutil
import math
import time
from pathlib import Path

dirname = os.path.dirname(os.path.abspath(__file__))  # Absolute path of this file

verboseLevel = 'error'  # detail of the logs (error, info, etc)

baseDir = ""


def SilentMkdir(theDir):    # function to create a directory
    try:
        os.mkdir(theDir)
    except:
        pass
    return 0


def run_1_cameraInit(binPath,baseDir,imgDir):

    taskFolder = '/1_CameraInit'
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 1/14 CAMERA INITIALIZATION -----------------------")

    imageFolder = imgDir + "/"
    sensorDatabase = str(Path(binPath).parent) + '/share/aliceVision/cameraSensors.db' # Path to the sensors database, might change in later versions of meshroom
   
    output = baseDir + taskFolder + '/cameraInit.sfm'

    cmdLine = binPath + "aliceVision_cameraInit"
    cmdLine += " --imageFolder {0} --sensorDatabase {1} --output {2}".format(
        imageFolder, sensorDatabase, output)

    cmdLine += " --defaultFieldOfView 45" 
    cmdLine += " --allowSingleView 1"
    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)

    return 0


def run_2_featureExtraction(binPath,baseDir , numberOfImages , imagesPerGroup=40, describerdensity="ultra",describerquality="ultra"):

    taskFolder = "/2_FeatureExtraction"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 2/14 FEATURE EXTRACTION -----------------------")

    _input = baseDir + '/1_CameraInit/cameraInit.sfm'
    output = baseDir + taskFolder + "/"

    cmdLine = binPath + "aliceVision_featureExtraction"
    cmdLine += " --input {0} --output {1}".format(_input, output)
    cmdLine += " --forceCpuExtraction 0"


    #when there are more than 40 images, it is good to send them in groups
    if(numberOfImages>imagesPerGroup):
        numberOfGroups=int(math.ceil( numberOfImages/imagesPerGroup))
        for i in range(numberOfGroups):
            cmd=cmdLine + " --rangeStart {} --rangeSize {} ".format(i*imagesPerGroup,imagesPerGroup)
            print("------- group {} / {} --------".format(i+1,numberOfGroups))
            print(cmd)
            os.system(cmd)

    else:
        print(cmdLine)
        os.system(cmdLine)


def run_3_imageMatching(binPath,baseDir):

    taskFolder = "/3_ImageMatching"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 3/14 IMAGE MATCHING -----------------------")

    _input = baseDir + '/1_CameraInit/cameraInit.sfm'
    featuresFolders = baseDir + '/2_FeatureExtraction' + "/"
    output = baseDir + taskFolder + '/imageMatches.txt'

    cmdLine = binPath + "aliceVision_imageMatching"
    cmdLine += " --input {0} --featuresFolders {1} --output {2}".format(
        _input, featuresFolders, output)

    cmdLine +=  " --tree " + "/"+ str(Path(binPath).parent)+ "/share/aliceVision/vlfeat_K80L3.SIFT.tree/"
    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)


def run_4_featureMatching(binPath,baseDir,numberOfImages,imagesPerGroup=20):

    taskFolder = "/4_featureMatching"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 4/14 FEATURE MATCHING -----------------------")

    _input =   baseDir + '/1_CameraInit/cameraInit.sfm'
    output =  baseDir + taskFolder + "/"
    featuresFolders =  baseDir + '/2_FeatureExtraction' + "/"
    imagePairsList =  baseDir + '/3_ImageMatching/imageMatches.txt'

    cmdLine = binPath + "aliceVision_featureMatching"
    cmdLine += " --input {0} --featuresFolders {1} --output {2} --imagePairsList {3}".format(
        _input, featuresFolders, output, imagePairsList)

    cmdLine += " --knownPosesGeometricErrorMax 5"
    cmdLine += " --verboseLevel " + verboseLevel

    cmdLine += " --describerTypes sift --photometricMatchingMethod ANN_L2 --geometricEstimator acransac --geometricFilterType fundamental_matrix --distanceRatio 0.8"
    cmdLine += " --maxIteration 2048 --geometricError 0.0 --maxMatches 0"
    cmdLine += " --savePutativeMatches False --guidedMatching False --matchFromKnownCameraPoses False --exportDebugFiles True"

    #when there are more than 20 images, it is good to send them in groups
    if(numberOfImages>imagesPerGroup):
        numberOfGroups=math.ceil( numberOfImages/imagesPerGroup)
        for i in range(numberOfGroups):
            cmd=cmdLine + " --rangeStart {} --rangeSize {} ".format(i*imagesPerGroup,imagesPerGroup)
            print("------- group {} / {} --------".format(i,numberOfGroups))
            print(cmd)
            os.system(cmd)

    else:
        print(cmdLine)
        os.system(cmdLine)

def run_5_structureFromMotion(binPath,baseDir):

    taskFolder = "/5_structureFromMotion"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 5/14 STRUCTURE FROM MOTION -----------------------")

    _input = baseDir + '/1_CameraInit/cameraInit.sfm'
    output = baseDir + taskFolder + '/sfm.abc' 
    outputViewsAndPoses = baseDir + taskFolder + '/cameras.sfm' 
    extraInfoFolder = baseDir + taskFolder + "/"
    featuresFolders = baseDir + '/2_FeatureExtraction' + "/"
    matchesFolders =  baseDir + '/4_featureMatching' + "/"

    cmdLine = binPath + "aliceVision_incrementalSfM"
    cmdLine += " --input {0} --output {1} --outputViewsAndPoses {2} --extraInfoFolder {3} --featuresFolders {4} --matchesFolders {5}".format(
        _input, output, outputViewsAndPoses, extraInfoFolder, featuresFolders, matchesFolders)

    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)


def run_6_prepareDenseScene(binPath,baseDir):
    taskFolder = "/6_PrepareDenseScene"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 6/14 PREPARE DENSE SCENE -----------------------")
    _input = baseDir +  '/5_structureFromMotion/sfm.abc'
    output = baseDir + taskFolder + "/" 

    cmdLine = binPath + "aliceVision_prepareDenseScene"
    cmdLine += " --input {0}  --output {1} ".format(_input,  output)

    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)


def run_7_depthMap(binPath,baseDir ,numberOfImages , groupSize=6 , downscale = 2):
    taskFolder = "/7_DepthMap"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 7/14 DEPTH MAP -----------------------")
    _input = baseDir +   '/5_structureFromMotion/sfm.abc'
    output = baseDir + taskFolder + "/"
    imagesFolder = baseDir + '/6_PrepareDenseScene' + "/"

    cmdLine = binPath + "aliceVision_depthMapEstimation"
    cmdLine += " --input {0}  --output {1} --imagesFolder {2}".format(
        _input,  output, imagesFolder)

    cmdLine += " --verboseLevel " + verboseLevel
    cmdLine += " --downscale " + str(downscale)
    
    numberOfBatches = int(math.ceil( numberOfImages / groupSize ))

    for i in range(numberOfBatches):
        groupStart = groupSize * i
        currentGroupSize = min(groupSize,numberOfImages - groupStart)
        if groupSize > 1:
            print("DepthMap Group {} of {} : {} to {}".format(i, numberOfBatches, groupStart, currentGroupSize))
            cmd = cmdLine + (" --rangeStart {} --rangeSize {}".format(str(groupStart),str(groupSize)))       
            print(cmd)
            os.system(cmd)


def run_8_depthMapFilter(binPath,baseDir):
    taskFolder = "/8_DepthMapFilter"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 8/14 DEPTH MAP FILTER-----------------------")
    _input = baseDir +   '/5_structureFromMotion/sfm.abc'
    output = baseDir + taskFolder + "/"
    depthMapsFolder = baseDir + '/7_DepthMap' + "/"

    cmdLine = binPath + "aliceVision_depthMapFiltering"
    cmdLine += " --input {0}  --output {1} --depthMapsFolder {2}".format(
        _input,  output, depthMapsFolder)

    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)


def run_9_meshing(binPath,baseDir  , maxInputPoints = 500000000  , maxPoints=100000000,colorizeoutput="True"):
    taskFolder = "/9_Meshing"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 9/14 MESHING -----------------------")
    _input = baseDir +  '/5_structureFromMotion/sfm.abc'
    output = baseDir +   taskFolder + '/densePointCloud.abc'
    outputMesh = baseDir + taskFolder + '/mesh.obj' 
    depthMapsFolder = baseDir + '/8_DepthMapFilter' + "/"

    cmdLine = binPath + "aliceVision_meshing"
    cmdLine += " --input {0}  --output {1} --outputMesh {2} --depthMapsFolder {3} ".format(
        _input,  output, outputMesh, depthMapsFolder)

    cmdLine += " --maxInputPoints " + str(maxInputPoints)
    cmdLine += " --maxPoints " + str(maxPoints)
    cmdLine += " --verboseLevel " + verboseLevel
    cmdLine += " --colorizeOutput " + colorizeoutput


    print(cmdLine)
    os.system(cmdLine)

def run_14_convertSFMFormat(binPath,baseDir  , SFMFileFormat = "ply"):
    taskFolder = "/14_convertSFMFormat"
    describers = "unknown"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 14/14 CONVERTING -----------------------")
    _input = baseDir +   '/9_Meshing/densePointCloud.abc'
    output = baseDir +   taskFolder + '/densePointCloud.ply'

    cmdLine = binPath + "aliceVision_convertSfMFormat"
    cmdLine += " --input {0}  --output {1} ".format(
        _input,  output)

    cmdLine += " --verboseLevel " + verboseLevel

    cmdLine += " --describerTypes " + describers


    print(cmdLine)
    os.system(cmdLine)


def run_10_meshFiltering(binPath,baseDir ,keepLargestMeshOnly="True", smoothingiterations=100):
    taskFolder = "/10_MeshFiltering"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 10/14 MESH FILTERING -----------------------")
    inputMesh = baseDir + '/9_Meshing/mesh.obj' 
    outputMesh = baseDir + taskFolder + '/mesh.obj'

    cmdLine = binPath + "aliceVision_meshFiltering"
    cmdLine += " --inputMesh {0}  --outputMesh {1}".format(
        inputMesh, outputMesh)

    cmdLine += " --verboseLevel " + verboseLevel
    cmdLine += " --keepLargestMeshOnly " + keepLargestMeshOnly

    print(cmdLine)
    os.system(cmdLine)


def run_11_meshDecimate(binPath,baseDir , simplificationFactor=0.0 , maxVertices=100000):
    taskFolder = "/11_MeshDecimate"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 11/14 MESH DECIMATE -----------------------")
    inputMesh = baseDir + '/10_MeshFiltering/mesh.obj'
    outputMesh = baseDir + taskFolder + '/mesh.obj'

    cmdLine = binPath + "aliceVision_meshDecimate"
    cmdLine += " --input {0}  --output {1}".format(
        inputMesh, outputMesh)

    cmdLine += " --verboseLevel " + verboseLevel
    cmdLine += " --simplificationFactor " + str(simplificationFactor)
    cmdLine += " --maxVertices " + str(maxVertices)

    print(cmdLine)
    os.system(cmdLine)


def run_12_meshResampling(binPath,baseDir , simplificationFactor=0.0 , maxVertices=100000):
    taskFolder = "/12_MeshResampling"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 12/14 MESH RESAMPLING -----------------------")
    inputMesh = baseDir +  '/11_MeshDecimate/mesh.obj' 
    outputMesh = baseDir  + taskFolder + '/mesh.obj'

    cmdLine = binPath + "aliceVision_meshResampling"
    cmdLine += " --input {0}  --output {1}".format( inputMesh, outputMesh)

    cmdLine += " --verboseLevel " + verboseLevel
    cmdLine += " --simplificationFactor " + str(simplificationFactor)
    cmdLine += " --maxVertices " + str(maxVertices)

    print(cmdLine)
    os.system(cmdLine)




def run_13_texturing(binPath , baseDir , textureSide = 16384 , downscale=1 , unwrapMethod = "Basic", fillholes="True"):
    taskFolder = '/13_Texturing'
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 13/14 TEXTURING  -----------------------")
    _input = baseDir +   '/9_Meshing/densePointCloud.abc' 
    imagesFolder = baseDir + '/6_PrepareDenseScene' "/"
    inputMesh = baseDir + '/12_MeshResampling/mesh.obj'
    output = baseDir + taskFolder + "/"

    cmdLine = binPath + "aliceVision_texturing"
    cmdLine += " --input {0} --inputMesh {1} --output {2} --imagesFolder {3}".format(
        _input, inputMesh, output, imagesFolder)

    cmdLine += " --textureSide " + str(textureSide)
    cmdLine += " --downscale " + str(downscale)
    cmdLine += " --verboseLevel " + verboseLevel
    cmdLine += " --unwrapMethod " + unwrapMethod

    print(cmdLine)
    os.system(cmdLine)


# Add ImageMatchingMultiSfM for recursive update 


def main():

    first_iteration_ = True
    global baseDir

    # Pass the arguments of the function as parameters in the command line code
    binPath = sys.argv[1]           ##  --> path of the binary files from Meshroom
    baseDir = sys.argv[2]           ##  --> name of the Folder containing the process (a new folder will be created)
    imgDir = sys.argv[3]            ##  --> Folder containing the images 
    rerunCylce = 5 ## --> Reconstruction rerun every ... number of new images

    SilentMkdir(baseDir)

    try:
        while True:

            startTime = time.time()

            numberOfImages = len([name for name in os.listdir(imgDir) if os.path.isfile(os.path.join(imgDir, name))])

            if first_iteration_:
                if numberOfImages > 5:
                    print("Images found in the directory. Starting Meshroom processing.")
                    updateProcess(binPath, imgDir, numberOfImages)
                    endTime = time.time()
                    hours, rem = divmod(endTime-startTime, 3600)
                    minutes, seconds = divmod(rem, 60)
                    print("time elapsed: "+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
                first_iteration_ = False
            elif checkForNewImages(numberOfImages, rerunCylce):
                print("New images detected. Updating the Meshroom processing steps.")
                updateProcess(binPath, imgDir, numberOfImages)
                endTime = time.time()
                hours, rem = divmod(endTime-startTime, 3600)
                minutes, seconds = divmod(rem, 60)
                print("time elapsed: "+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
            else:
                print("No new images detected. Waiting for changes...")
                time.sleep(5)  # 5 seconds wait for new check
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")

def checkForNewImages(currentCount, rerunLimit):
    global baseDir
    lastCountFile = os.path.join(baseDir, "last_image_count.txt")

    if os.path.exists(lastCountFile):
        with open(lastCountFile, "r") as f:
            lastCount = int(f.read().strip())
        if currentCount >= lastCount + rerunLimit:
            with open(lastCountFile, "w") as f:
                f.write(str(currentCount))
            return True
        else:
            return False
    else:
        # First run, create the last count file
        with open(lastCountFile, "w") as f:
            f.write(str(currentCount))
        return False

def updateProcess(binPath, imgDir, numberOfImages):

    run_1_cameraInit(binPath,baseDir,imgDir)
    run_2_featureExtraction(binPath,baseDir, numberOfImages)
    run_3_imageMatching(binPath,baseDir)
    run_4_featureMatching(binPath,baseDir,numberOfImages)
    run_5_structureFromMotion(binPath,baseDir)
    run_6_prepareDenseScene(binPath,baseDir)
    run_7_depthMap(binPath,baseDir , numberOfImages )
    run_8_depthMapFilter(binPath,baseDir)
    run_9_meshing(binPath,baseDir)
    #run_10_meshFiltering(binPath,baseDir)
    #run_11_meshDecimate(binPath,baseDir)
    #run_12_meshResampling(binPath,baseDir)
    #run_13_texturing(binPath,baseDir)
    run_14_convertSFMFormat(binPath,baseDir)


if __name__ == "__main__":
    main()
