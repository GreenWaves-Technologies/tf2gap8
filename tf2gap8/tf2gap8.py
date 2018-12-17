'''/*
* Copyright (c) 2017 GreenWaves Technologies SAS
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
* 3. Neither the name of GreenWaves Technologies SAS nor the names of its
*    contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/
/*
This script automates the generation of GAP8 processor source code for 
a Convolutional Neural Network (CNN) application described using the 
Tensorflow r1.2 API.
This script input is the Tensorflow description of a CNN application. Its output is
the corresponding  representation in GAP8 C/C++ source code using the GAP8 CNN and 
operators library. The steps that this script automates are the following ones:
    * Training of the TensorFlow CNN application written with TF API and Python
    * Using the TF Graph Transform Tool (GTT) to transform the graph and prepare 
      it for the inference phase on GAP8 processor. The main tasks of this 
      transformation will be to:
        * Trim parts of the graph that are not needed for inference
        * Turn sub-expressions into single nodesquant
        * Order nodes in processing order for inference
        * Refactor some nodes to fit the GAP8 CNN functions Library
    * Inference code generation from the application graph that has been previously 
      transformed, using GAP8 CNN and operators library
*/
'''

# Import the necessary modules
import subprocess
import sys
import getopt

# Define some utility functions

def getGraphDir(inputGraph):
    words=inputGraph.split('/')
    graphDir=''
    for i in range(0,len(words) -1):
        if (i !=len(words)-1):
            if (words[i]!=''):
                graphDir=graphDir + '/' + words[i]
    return graphDir

def getGraphName(inputGraph):
    words=inputGraph.split('/')
    name=words[len(words)-1]
    words=name.split('.')
    return (words[0])


def getFrozenGraphName(inputGraph):
    graphDir=getGraphDir(inputGraph)
    frozenGraph=graphDir + "/" + getGraphName(inputGraph) + "_frozen.pb"
    return frozenGraph

def getOptimizedGraphName(inputGraph, runExtension=""):
    graphDir=getGraphDir(inputGraph)
    optimizedGraph=graphDir + "/" + getGraphName(inputGraph) + "_optimized" + runExtension + ".pb"
    return optimizedGraph

def usage():
    print( 'tf2gap8.py --input_graph <input_graph_file> --input_checkpoint <checkpoint_file>' + ' --input_node <node name> --output_node <node name>\n' )


def removeFile(fileName):
    #print("**** Removing temporary files :"  + fileName + "   ******")
    result = subprocess.run(["rm", fileName], stdout=subprocess.PIPE)
    result.stdout.decode('utf-8')


def main(argv):
    #Main script function
    #usage 
    #tf2gap8.py --input_graph <input_graph_file> --input_checkpoint <checkpoint_file>' + ' --input_node <node name> --output_node <node name> --floating_point <boolean>
    #-h : is the help option

    print('\x1b[6;30;42m' + 'Begin tf2gap8 bridge script' + '\x1b[0m')

    #Initializing some variables
    inputGraph=''
    inputCheckpoint=''
    inputNode=''
    TFDir="/home/corine_lamagdeleine/tensorflow"
    #Parameters deducted from previous ones
    frozenGraphName=''
    ftp=''
    freeze=''
   
    # Get options and their value
    print("Starting tf2gap8.py Script\n")
    try:

        print( 'Number of arguments: ' + str(len(sys.argv) )+ ' arguments.\n')
        print( 'Argument List: ' + str(sys.argv))
        #First argument is the python file name itself
        #We will have the following options:
        # --input_graph=/home/corine_lamagdeleine/tmp2/cifar10_train/graph.pbtxt
        # --input_checkpoint=/home/corine_lamagdeleine/tmp2/cifar10_train/model.ckpt-10
        # not needed: --output_graph=/home/corine_lamagdeleine//tmp2/cifar10_train/frozen_graph.pb
        # --output_node_names=softmax_linear/softmax_linear
        # --tf_dir
        # --floating_point
        # --freeze

        opts, args = getopt.getopt(argv,"h", ["help","input_graph=","input_checkpoint=","input_node=","output_node=", "tf_dir=","floating_point=","freeze="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt=='--help':
            usage()
            sys.exit()
        elif opt =="--input_graph":
            inputGraph = arg
        elif opt=="--input_checkpoint":
            inputCheckpoint = arg
        elif opt=="--input_node":
            inputNode=arg
        elif opt=="--output_node":
            outputNode=arg
        elif opt=="--tf_dir":
            TFDir=arg
        elif opt=="--floating_point":
            ftp=arg
        elif opt=="--freeze":
            freeze=arg
        else:
            assert False, "unhandled option"

    if (freeze=='true'):
        #input graph is not already frozen. It has to be freezed
        #generate frozen Graph File name
        frozenGraphName=getFrozenGraphName(inputGraph)
    else:
        # input graph is already frozen
        frozenGraphName=inputGraph

    print('Input Graph is: ' + inputGraph +"\n")
    print('Input Checkpoint is: ' + inputCheckpoint +"\n")
    print('Input Node is: ' + inputNode+"\n")
    print('Output Node is: ' + outputNode+"\n")
    print('Tensorflow Dir is: ' + TFDir +"\n")
    print('floating_point is: ' + ftp + "\n")
    print("freeze is:" + freeze + "\n")
    print("frozenGraphName: " + frozenGraphName + "\n")


    if (freeze=='true'):
        #Run freeze graph command to freeze the TF graph with its correspondig weights and biases
        #resulting from the training step
        print('\x1b[6;30;42m' + 'Calling subprocess freeze_graph' + '\x1b[0m')
        print ("\n")
        print(inputGraph)
        print(inputCheckpoint)
        print(getFrozenGraphName(inputGraph))
        print(inputNode)
        print(outputNode)
        print(TFDir+ '/bazel-bin/tensorflow/python/tools/freeze_graph ' + '--input_graph=' + inputGraph + ' --input_checkpoint=' + inputCheckpoint + ' --output_graph=' + getFrozenGraphName(inputGraph) + ' --output_node_names=' + outputNode)
        result = subprocess.run([TFDir + '/bazel-bin/tensorflow/python/tools/freeze_graph',
                                '--input_graph=' + inputGraph,
                                '--input_checkpoint=' + inputCheckpoint,
                                '--output_graph=' + getFrozenGraphName(inputGraph),
                                '--output_node_names=' + outputNode],
                                stdout=subprocess.PIPE)
        '''result.stdout.decode('utf-8')
        print (result)'''
        returncode=result.returncode
        if (returncode!=0):
            print ('\x1b[250;4;4m' + "freeze_graph process didn't complete" + '\x1b[0m')
            quit()
        else:
            print ('\x1b[6;30;42m' + "Subprocess freeze_graph completed successfully" + '\x1b[0m')
        print ("\n")

    # Now transform the graph obtained. Frozen graph has been stored in the same directory as the original graph.
    ''''result=applyAllTransforms( getFrozenGraphName(inputGraph), 
                        getOptimizedGraphName(inputGraph),
                        TFDir,inputNode, outputNode)'''
    print('\x1b[6;30;42m' + 'Calling subprocess transform_graph' + '\x1b[0m')
    print ("\n")
    '''returncode=applyAllTransforms( getFrozenGraphName(inputGraph), 
                        getOptimizedGraphName(inputGraph),
                        TFDir,inputNode, outputNode)'''
    
    print('\x1b[6;30;42m' + 'Calling subprocess transform_graph' + '\x1b[0m')
    print('--transforms=strip_unused_nodes remove_nodes(op=Identity) fuse_conv2d_add_relu_maxpool fuse_conv2d_add_relu fuse_conv2d_add_maxpool fuse_GAP8_conv2d_maxpool fuse_reshape_matmul_add_relu_softmax fuse_reshape_matmul_add_softmax')
    print(TFDir + '/bazel-bin/tensorflow/tools/graph_transforms/transform_graph' + ' --in_graph='+ frozenGraphName + ' --out_graph=' + getOptimizedGraphName(inputGraph) + ' --inputs=' + inputNode + ' --outputs='+ outputNode + ' --transforms=strip_unused_nodes gap8_transform_tool remove_nodes(op=Identity)')
    result=subprocess.run([TFDir +'/bazel-bin/tensorflow/tools/graph_transforms/transform_graph',
                          '--in_graph='+ frozenGraphName,
                          '--out_graph=' + getOptimizedGraphName(inputGraph),
                          '--inputs=' + inputNode,
                          '--outputs=' + outputNode,
                          '--transforms=strip_unused_nodes remove_nodes(op=Identity) fuse_conv2d_add_relu_maxpool fuse_conv2d_add_relu fuse_conv2d_add_maxpool fuse_GAP8_conv2d_maxpool fuse_reshape_matmul_add_relu_softmax fuse_reshape_matmul_add_softmax'], stdout=subprocess.PIPE)
    
    #Now generate the GAP8 code from the optimized graph
    #This program now called loader.cc must be called with the file name (without path) and #directory name as parameters of the file
    returncode=result.returncode
    if (returncode!=0):
        print ('\x1b[250;4;4m' + "transform_graph process didn't complete" + '\x1b[0m')
        quit()
    else:
        print ('\x1b[6;30;42m' + "Subprocess transform_graph completed successfully" + '\x1b[0m')
    print ("\n")

    print('\x1b[6;30;42m' + 'Calling tf2gap8' + '\x1b[0m')
    print ("\n")
    result=subprocess.run([TFDir + '/bazel-bin/tf2gap8/tf2gap8', getGraphName(inputGraph) + "_optimized" + ".pb",
                           getGraphDir(inputGraph),TFDir + '/tf2gap8',ftp],stdout=subprocess.PIPE)
    returncode=result.returncode
    if (returncode!=0):
        print ('\x1b[250;4;4m' + "tf2gap8 process didn't complete" + '\x1b[0m')
        quit()
    else:
        print ('\x1b[6;30;42m' + "tf2gap8 process completed successfully" + '\x1b[0m')
    print ("\n")

    if (returncode!=0):
        print ('\x1b[250;4;4m' + "tf2gap8 bridge script didn't complete successfully" + '\x1b[0m')
    else:
        print('\x1b[6;30;42m' + 'tf2gap8 bridge script completed successfully' + '\x1b[0m')
    result.stdout.decode('utf-8')

if __name__ == "__main__":
    main(sys.argv[1:])
