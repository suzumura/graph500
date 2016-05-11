#! /usr/bin/env python

from optparse import OptionParser

# parse command line arguments

parser = OptionParser()
parser.add_option("-t","--threads", action="store", dest="numThreads", default=4, type="int")
parser.add_option("-s","--scale", action="store", dest="numScale", default=16, type="int")
parser.add_option("-v","--verbose", action="store_true", dest="verboseMode", default=False)
parser.add_option("--vertex-reordering", action="store", dest="vertexReordering", type="int", default=0)
parser.add_option("-b", "--bind-mode", action="store", dest="bindMode", type="choice", choices=["NONE", "TRUE", "SPREAD", "CLOSE"], default="NONE")

(options, args) = parser.parse_args()

# confirmation

print("Number of Scale : " + str(options.numScale))
print("Number of Threads : " + str(options.numThreads))
print("Verbose Mode : " + str(options.verboseMode))
print("Vertex Reordering Mode : " + str(options.vertexReordering))
print("Thread Binding Mode : " + str(options.bindMode))
