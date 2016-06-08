#! /usr/bin/env python

from optparse import OptionParser
import subprocess

#####################################
# mpirun command spawning functions
#####################################

# log file name generator
def generateLogFileName(options):
    outFilePattern = "lP1"
    outFilePattern += "T" + str(options.numThreads)
    outFilePattern += "S" + str(options.numScale)
    outFilePattern += "VR" + str(options.vertexReordering)
    outFilePattern += "B" + str(options.bindMode)
    return outFilePattern

# for OpenMPI
def spawnOpenMPI(options):
    benchArgs = ["mpirun", "-np", "1", "-bind-to", "none"]

    outFilePattern = generateLogFileName(options)
    benchArgs.extend(["-output-filename", options.logfileDestination + "/" + outFilePattern])

    benchArgs.extend(["-x", "OMP_NUM_THREADS=" + str(options.numThreads)])

    if options.bindMode != "NONE":
        benchArgs.extend(["-x", "OMP_PROC_BIND=" + options.bindMode])

    benchArgs.extend(["./mpi/runnable", str(options.numScale)])

    returnCode = 0
    if options.printCommandMode:
        print(" ".join(benchArgs))
    else:
        benchProc = subprocess.Popen(benchArgs)
        benchProc.communicate()
        returnCode = benchProc.returncode
        del benchProc

    return returnCode

# for OpenMPI (<= version 1.6.5)
def spawnOpenMPIOld(options):
    benchArgs = ["mpirun", "-np", "1", "-bind-to-none"]

    outFilePattern = generateLogFileName(options)
    benchArgs.extend(["-output-filename", options.logfileDestination + "/" + outFilePattern])

    benchArgs.extend(["-x", "OMP_NUM_THREADS=" + str(options.numThreads)])

    if options.bindMode != "NONE":
        benchArgs.extend(["-x", "OMP_PROC_BIND=" + options.bindMode])

    benchArgs.extend(["./mpi/runnable", str(options.numScale)])

    returnCode = 0
    if options.printCommandMode:
        print(" ".join(benchArgs))
    else:
        benchProc = subprocess.Popen(benchArgs)
        benchProc.communicate()
        returnCode = benchProc.returncode
        del benchProc

    return returnCode

# for MPICH/MVAPICH
def spawnMPICH(options):
    benchArgs = ["mpirun", "-n", "1"]

    outFilePattern = generateLogFileName(options)
    benchArgs.extend(["-outfile-pattern", options.logfileDestination + "/" + outFilePattern])

    benchArgs.extend(["-genv", "OMP_NUM_THREADS", str(options.numThreads)])

    if options.bindMode != "NONE":
        benchArgs.extend(["-genv", "OMP_PROC_BIND", options.bindMode])

    benchArgs.extend(["./mpi/runnable", str(options.numScale)])

    returnCode = 0
    if options.printCommandMode:
        print(" ".join(benchArgs))
    else:
        benchProc = subprocess.Popen(benchArgs)
        benchProc.communicate()
        returnCode = benchProc.returncode
        del benchProc

    return returnCode


# parse command line arguments

parser = OptionParser()
parser.add_option("-t","--threads", action="store", dest="numThreads", default=4, type="int")
parser.add_option("-s","--scale", action="store", dest="numScale", default=16, type="int")
parser.add_option("-v","--verbose", action="store_true", dest="verboseMode", default=False)
parser.add_option("--vertex-reordering", action="store", dest="vertexReordering", type="int", default=0)
parser.add_option("-b", "--bind-mode", action="store", dest="bindMode", type="choice", choices=["NONE", "TRUE", "SPREAD", "CLOSE"], default="NONE")
parser.add_option("--test", action="store_true", dest="testMode", default=False)
parser.add_option("--increasing-scale", action="store_true", dest="increasingScale", default=False)
parser.add_option("--logfile-dest", action="store", dest="logfileDestination", default="./log", type="string")
parser.add_option("--print-command", action="store_true", dest="printCommandMode", default=False)
parser.add_option("-m", "--mpi-runtime", action="store", dest="mpiRuntime", type="choice", choices=["OpenMPI", "OpenMPIOld", "MPICH", "MVAPICH"], default="MPICH")

(options, args) = parser.parse_args()

# confirmation

if not options.printCommandMode:
    print("Number of Scale : " + str(options.numScale))
    print("Number of Threads : " + str(options.numThreads))
    print("Verbose Mode : " + str(options.verboseMode))
    print("Vertex Reordering Mode : " + str(options.vertexReordering))
    print("Thread Binding Mode : " + str(options.bindMode))
    print("Test Mode : " + str(options.testMode))


# rebuild benchmark program

buildArgs = ["make"]

## set verbose option
if options.verboseMode:
    buildArgs.append("VERBOSE=true")

## set vertex reordering options
buildArgs.append(" VERTEX_REORDERING=" + str(options.vertexReordering))

## set state whether benchmarking or testing
if not options.testMode:
    buildArgs.append("REAL_BENCHMARK=true")

## append build target
buildArgs.append("cpu")

## compare previous build options and this time's
sameBuildOptions = False
try:
    prevBuildOptFile = open("prev_build_options", "r")
    prevBuildOptStr = prevBuildOptFile.readline()
    if prevBuildOptStr == str(buildArgs):
        sameBuildOptions = True
    else:
        sameBuildOptions = False
        prevBuildOptFile.close()
        prevBuildOptFile = open("prev_build_options", "w")
        prevBuildOptFile.write(str(buildArgs))
    prevBuildOptFile.close()
except IOError:
    sameBuildOptions = False
    prevBuildOptFile = open("prev_build_options", "w")
    prevBuildOptFile.write(str(buildArgs))

#print(str(sameBuildOptions))

## rebuild if nesessary
if not sameBuildOptions:
    if options.printCommandMode:
        print("cd mpi")
        print("make clean")
        print(" ".join(buildArgs))
        print("cd ..")
    else:
        buildProc = subprocess.Popen(["make", "clean"], cwd="mpi")
        buildProc.communicate()
        buildProc = subprocess.Popen(buildArgs, cwd="mpi")
        buildProc.communicate()

# run benchmark
commandFunction = None
if options.mpiRuntime == "OpenMPI":
    commandFunction = spawnOpenMPI
elif options.mpiRuntime == "OpenMPIOld":
    commandFunction = spawnOpenMPIOld
elif options.mpiRuntime == "MPICH" or options.mpiRuntime == "MVAPICH":
    commandFunction = spawnMPICH

while True:

    returnCode = commandFunction(options)
    if options.increasingScale and returnCode == 0 and not options.printCommandMode:
        options.numScale += 1
	print("continueing... scale : " + str(options.numScale))
    else:
        print("return code: " + str(returnCode))
        break
