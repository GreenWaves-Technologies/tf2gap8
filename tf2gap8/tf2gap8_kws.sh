#!/bin/bash
'''
*
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
'''

# Script Shell for running the speech train example 

printf "****** Freeze graph ******\n"
python3 $HOME/tensorflow/tf2gap8/examples/kws/freeze.py --start_checkpoint=$HOME/tensorflow/tf2gap8/examples/kws/data/conv.ckpt-18000  --output_file=$HOME/tensorflow/tf2gap8/examples/kws/data/my_frozen_graph.pb
printf "****** Freeze Graph Ended ******\n"
printf "****** Transform Graph ******\n"
$HOME/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph --in_graph=$HOME/tensorflow/tf2gap8/examples/kws/data/my_frozen_graph.pb --out_graph=$HOME/tensorflow/tf2gap8/examples/kws/data/optimized_graph.pb --inputs="Maxpool" --outputs="add_2" --transforms="strip_unused_nodes remove_nodes(op=Identity) fuse_conv2d_add_relu_maxpool fuse_conv2d_add_relu fuse_conv2d_add_maxpool fuse_GAP8_conv2d_maxpool fuse_reshape_matmul_add_relu_softmax fuse_reshape_matmul_add_softmax fuse_reshape_matmul_add"
printf "****** Transform Graph Ended ******\n"
printf "****** TF2GAP8 ******\n"
$HOME/tensorflow/bazel-bin/tf2gap8/tf2gap8 optimized_graph.pb $HOME/tensorflow/tf2gap8/examples/kws/data $HOME/tensorflow/tf2gap8 true

printf "****** TF2GAP8 Ended******\n"

