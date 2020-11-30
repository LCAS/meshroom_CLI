import sys
import os
import shutil
import math

dirname = os.path.dirname(os.path.abspath(__file__))  # Absolute path

# path of the bin files of meshroom
# binPath = "Meshroom-2020.1.1\\aliceVision\\bin"
# baseDir = "intermediate"  # path of the output and temporal files
# # path of the folder containing the images
# imgDir = "dataset_monstree-master/mini6"
verboseLevel = "\"" + "error" + "\""  # detail of the logs (error, info, etc)


def SilentMkdir(theDir):    # function to create a directory
    try:
        os.mkdir(theDir)
    except:
        pass
    return 0


def run_1_cameraInit(binPath,baseDir,imgDir):

    taskFolder = "/1_CameraInit"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 1/11 CAMERA INITIALIZATION -----------------------")

    imageFolder = "\"" + imgDir + "\""
    sensorDatabase = "\"\""
    output = "\"" + baseDir + taskFolder + "/cameraInit.sfm" + "\""

    cmdLine = binPath + "\\aliceVision_cameraInit.exe"
    cmdLine += " --imageFolder {0} --sensorDatabase {1} --output {2}".format(
        imageFolder, sensorDatabase, output)

    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)

    return 0


def run_2_featureExtraction(binPath,baseDir):

    taskFolder = "/2_FeatureExtraction"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 2/11 FEATURE EXTRACTION -----------------------")

    _input = "\"" + baseDir + "/1_CameraInit/cameraInit.sfm" + "\""
    output = "\"" + baseDir + taskFolder + "\""

    cmdLine = binPath + "\\aliceVision_featureExtraction.exe"
    cmdLine += " --input {0} --output {1} ".format(_input, output)

    print(cmdLine)
    os.system(cmdLine)


def run_3_imageMatching(binPath,baseDir):

    taskFolder = "/3_ImageMatching"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 3/11 IMAGE MATCHING -----------------------")

    _input = "\"" + baseDir + "/1_CameraInit/cameraInit.sfm" + "\""
    featuresFolders = "\"" + baseDir + "/2_FeatureExtraction" + "\""
    output = "\"" + baseDir + taskFolder + "/imageMatches.txt" + "\""

    cmdLine = binPath + "\\aliceVision_imageMatching.exe"
    cmdLine += " --input {0} --featuresFolders {1} --output {2}".format(
        _input, featuresFolders, output)

    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)


def run_4_featureMatching(binPath,baseDir):

    taskFolder = "/4_featureMatching"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 4/11 FEATURE MATCHING -----------------------")

    _input = "\"" + baseDir + "/1_CameraInit/cameraInit.sfm" + "\""
    output = "\"" + baseDir + taskFolder + "\""
    featuresFolders = "\"" + baseDir + "/2_FeatureExtraction" + "\""
    imagePairsList = "\"" + baseDir + "/3_ImageMatching/imageMatches.txt" + "\""

    cmdLine = binPath + "\\aliceVision_featureMatching.exe"
    cmdLine += " --input {0} --featuresFolders {1} --output {2} --imagePairsList {3} ".format(
        _input, featuresFolders, output, imagePairsList)

    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)


def run_5_structureFromMotion(binPath,baseDir):

    taskFolder = "/5_structureFromMotion"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 5/11 STRUCTURE FROM MOTION -----------------------")

    _input = "\"" + dirname + "/" + baseDir + "/1_CameraInit/cameraInit.sfm" + "\""
    output = "\"" + dirname + "/" + baseDir + taskFolder + "/sfm.abc" + "\" "
    outputViewsAndPoses = "\"" + dirname + "/" + \
        baseDir + taskFolder + "/cameras.sfm" + "\""
    extraInfoFolder = "\"" + dirname + "/" + baseDir + taskFolder + "\""
    featuresFolders = "\"" + dirname + "/" + \
        baseDir + "/2_FeatureExtraction" + "\""
    matchesFolders = "\"" + dirname + "/" + baseDir + "/4_featureMatching" + "\""

    cmdLine = binPath + "\\aliceVision_incrementalSfm.exe"
    cmdLine += " --input {0} --output {1} --outputViewsAndPoses {2} --extraInfoFolder {3} --featuresFolders {4} --matchesFolders {5}".format(
        _input, output, outputViewsAndPoses, extraInfoFolder, featuresFolders, matchesFolders)

    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)


def run_6_prepareDenseScene(binPath,baseDir):
    taskFolder = "/6_PrepareDenseScene"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 6/11 PREPARE DENSE SCENE -----------------------")
    _input = "\"" + dirname + "/" + baseDir + \
        "/5_structureFromMotion/sfm.abc" + "\""
    output = "\"" + dirname + "/" + baseDir + taskFolder + "\" "

    cmdLine = binPath + "\\aliceVision_prepareDenseScene.exe"
    cmdLine += " --input {0}  --output {1} ".format(_input,  output)

    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)


def run_7_depthMap(binPath,baseDir):
    taskFolder = "/7_DepthMap"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 7/11 DEPTH MAP -----------------------")
    _input = "\"" + dirname + "/" + baseDir + \
        "/5_structureFromMotion/sfm.abc" + "\""
    output = "\"" + dirname + "/" + baseDir + taskFolder + "\""
    imagesFolder = "\"" + dirname + "/" + baseDir + "/6_PrepareDenseScene" + "\""

    cmdLine = binPath + "\\aliceVision_depthMapEstimation.exe"
    cmdLine += " --input {0}  --output {1} --imagesFolder {2}".format(
        _input,  output, imagesFolder)

    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)


def run_8_depthMapFilter(binPath,baseDir):
    taskFolder = "/8_DepthMapFilter"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 8/11 DEPTH MAP FILTER-----------------------")
    _input = "\"" + dirname + "/" + baseDir + \
        "/5_structureFromMotion/sfm.abc" + "\""
    output = "\"" + dirname + "/" + baseDir + taskFolder + "\""
    depthMapsFolder = "\"" + dirname + "/" + baseDir + "/7_DepthMap" + "\""

    cmdLine = binPath + "\\aliceVision_depthMapFiltering.exe"
    cmdLine += " --input {0}  --output {1} --depthMapsFolder {2}".format(
        _input,  output, depthMapsFolder)

    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)


def run_9_meshing(binPath,baseDir , maxInputPoints = 1000000  , maxPoints=2000000):
    taskFolder = "/9_Meshing"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 9/11 MESHING -----------------------")
    _input = "\"" + dirname + "/" + baseDir + \
        "/5_structureFromMotion/sfm.abc" + "\""
    output = "\"" + dirname + "/" + baseDir + \
        taskFolder + "/densePointCloud.abc" "\""
    outputMesh = "\"" + dirname + "/" + baseDir + taskFolder + "/mesh.obj" + "\""
    depthMapsFolder = "\"" + dirname + "/" + baseDir + "/8_DepthMapFilter" + "\""

    cmdLine = binPath + "\\aliceVision_meshing.exe"
    cmdLine += " --input {0}  --output {1} --outputMesh {2} --depthMapsFolder {3} ".format(
        _input,  output, outputMesh, depthMapsFolder)

    cmdLine += " --maxInputPoints " + str(maxInputPoints)
    cmdLine += " --maxPoints " + str(maxPoints)
    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)


def run_10_meshFiltering(binPath,baseDir ,keepLargestMeshOnly="True"):
    taskFolder = "/10_MeshFiltering"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 10/11 MESH FILTERING -----------------------")
    inputMesh = "\"" + dirname + "/" + baseDir + "/9_Meshing/mesh.obj" + "\""
    outputMesh = "\"" + dirname + "/" + baseDir + taskFolder + "/mesh.obj" + "\""

    cmdLine = binPath + "\\aliceVision_meshFiltering.exe"
    cmdLine += " --inputMesh {0}  --outputMesh {1}".format(
        inputMesh, outputMesh)

    cmdLine += " --verboseLevel " + verboseLevel
    cmdLine += " --keepLargestMeshOnly " + keepLargestMeshOnly

    print(cmdLine)
    os.system(cmdLine)








def run_11_texturing(binPath , baseDir , textureSide = 4096 , downscale=4):
    taskFolder = "/11_Texturing"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 11/11 TEXTURING  -----------------------")
    _input = "\"" + dirname + "/" + baseDir + \
        "/9_Meshing/densePointCloud.abc" + "\""
    imagesFolder = "\"" + dirname + "/" + baseDir + "/6_PrepareDenseScene" "\""
    inputMesh = "\"" + dirname + "/" + baseDir + "/10_MeshFiltering/mesh.obj" + "\""
    output = "\"" + dirname + "/" + baseDir + taskFolder + "\""

    cmdLine = binPath + "\\aliceVision_texturing.exe"
    cmdLine += " --input {0} --inputMesh {1} --output {2} --imagesFolder {3}".format(
        _input, inputMesh, output, imagesFolder)

    cmdLine += " --textureSide " + str(textureSide)
    cmdLine += " --downscale " + str(downscale)
    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)



def main():

    # Pass the arguments of the function as parameters in the command line code
    binPath = sys.argv[1]           ##  --> path of the binary files from Meshroom
    baseDir = sys.argv[2]           ##  --> name of the Folder containing the process (a new folder will be created)
    imgDir = sys.argv[3]            ##  --> Folder containing the images 


    SilentMkdir(baseDir)

    # run_1_cameraInit(binPath,baseDir,imgDir)
    # run_2_featureExtraction(binPath,baseDir)
    # run_3_imageMatching(binPath,baseDir)
    # run_4_featureMatching(binPath,baseDir)
    # run_5_structureFromMotion(binPath,baseDir)
    # run_6_prepareDenseScene(binPath,baseDir)
    # run_7_depthMap(binPath,baseDir)
    # run_8_depthMapFilter(binPath,baseDir)
    # run_9_meshing(binPath,baseDir)
    run_10_meshFiltering(binPath,baseDir)
    run_11_texturing(binPath,baseDir)
    print("-------------------------------- DONE ----------------------")
    input("press any key to close")


main()
