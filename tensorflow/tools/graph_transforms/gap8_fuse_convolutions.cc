/*
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

GAP_ GRAPH TRANSFORM TOOLS
Contributors: Nicolas Lepetit, Corine Lamagdeleine

In this first version of the TF2GAP8 tool, GreenWaves Technologies (GWT) supports the following CNN operations of Tensorflow: 

Add
Conv2D
Softmax
Matmul
Reshape
Relu

In order to match to the operators of the GAP8 CNN Library,  we have added some node factorisations to GTT under the option “gap8_transform_tool” of the command --transforms. During this processing, the following transformations occur:

ADD + conv2D --> GAP8_ConvLayer
conv2D + ADD + RELU  --> GAP8_convlayer_with_relu
Softmax + ADD + Matmul + Reshape  --> GAP8_DenseLayer
Softmax + RELU + ADD + Matmul + Reshape --> GAP8_DenseLayer_Relu
RELU + GAP8_ConvLayer --> GAP8_ConvLayer_Relu
*/

#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"
#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"
#include <iostream>
#include <fstream>

using namespace std;

namespace tensorflow {
namespace graph_transforms {

Status FuseConv2DAndAddAndReluAndMaxpool( const GraphDef& input_graph_def,
                                          const TransformFuncContext& context,
                                          GraphDef* output_graph_def) {

  // Keep looking for nodes to remove until there are no more changes.
  bool any_nodes_removed;
  GraphDef current_graph_def;
  current_graph_def=input_graph_def;
  do {
  any_nodes_removed=false;
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      current_graph_def, // clang-format of
      {"MaxPool",
        {
          {"Relu",
            {
              {"Add",
                {
                  {"Conv2D",
                    {
                      {"*"},
                      {"*"},
                    }
                  },
                  {"*"}
                }
              }
            }
          }
        }
      },  // clang-format on
      [&any_nodes_removed](
          const NodeMatch& match,
          const std::set<string>& input_nodes,
          const std::set<string>& output_nodes,
          std::vector<NodeDef>* new_nodes) {

            const NodeDef& maxpool_node = match.node;
            const NodeDef& add_node = match.inputs[0].inputs[0].node;
            CHECK_EQ("Add", add_node.op());
            const NodeDef& conv_node = match.inputs[0].inputs[0].inputs[0].node;
            CHECK_EQ("Conv2D", conv_node.op());
            const NodeDef& bias_node = match.inputs[0].inputs[0].inputs[1].node;
            const NodeDef& input_node = match.inputs[0].inputs[0].inputs[0].inputs[0].node;
            const NodeDef& weights_node = match.inputs[0].inputs[0].inputs[0].inputs[1].node;
            cerr << "++++++ FuseConv2DAndAddAndReluAndMaxpool +++++++++" << "\n";
            new_nodes->push_back(bias_node);
            new_nodes->push_back(input_node);
            new_nodes->push_back(weights_node);

            NodeDef conv;
            conv.set_op("GAP8_Conv2D");
            conv.set_name(match.node.name());
            AddNodeInput(conv_node.input(0), &conv);
            AddNodeInput(conv_node.input(1), &conv);
            AddNodeInput(add_node.input(1), &conv);
            CopyNodeAttr(add_node, "T", "T", &conv);
            CopyNodeAttr(conv_node, "padding", "padding", &conv);
            CopyNodeAttr(conv_node, "strides", "strides", &conv);
            CopyNodeAttr(maxpool_node, "padding", "maxpool_padding", &conv);
            CopyNodeAttr(maxpool_node, "strides", "maxpool_strides", &conv);
            CopyNodeAttr(maxpool_node,"ksize","pooling_factor",&conv);
            AddNodeAttr("maxpool",true,&conv);
            AddNodeAttr("relu",true,&conv);
            CopyNodeAttr(match.node, "_output_shapes", "_output_shapes", &conv);
            new_nodes->push_back(conv);
            any_nodes_removed=true;
            return Status::OK();
          },
          {}, &replaced_graph_def));
    current_graph_def=replaced_graph_def;       
   *output_graph_def = replaced_graph_def;

  } while (any_nodes_removed);
  return Status::OK();
}


Status FuseConv2DAndAddAndMaxpool( const GraphDef& input_graph_def,
                                  const TransformFuncContext& context,
                                  GraphDef* output_graph_def) {
bool any_nodes_removed;
GraphDef current_graph_def;
current_graph_def=input_graph_def;
do {
  GraphDef replaced_graph_def;
  any_nodes_removed=false;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      current_graph_def,  // clang-format of
      {"MaxPool",
        {
          {"Add",
            {
              {"Conv2D",
                {
                  {"*"},
                  {"*"},
                }
              },
              {"*"}
            }
          }
        }
      },  // clang-format on
      [&any_nodes_removed](
          const NodeMatch& match,
          const std::set<string>& input_nodes,
          const std::set<string>& output_nodes,
          std::vector<NodeDef>* new_nodes) {

            const NodeDef& add_node = match.inputs[0].node;
            CHECK_EQ("Add", add_node.op());
            const NodeDef& conv_node = match.inputs[0].inputs[0].node;
            CHECK_EQ("Conv2D", conv_node.op());
            const NodeDef& bias_node = match.inputs[0].inputs[1].node;
            const NodeDef& input_node = match.inputs[0].inputs[0].inputs[0].node;
            const NodeDef& weights_node = match.inputs[0].inputs[0].inputs[1].node;
            cerr << "++++++ FuseConv2DAndAddAndMaxpool +++++++++" << "\n"; 
            new_nodes->push_back(bias_node);
            new_nodes->push_back(input_node);
            new_nodes->push_back(weights_node);

            NodeDef conv_with_relu;
            conv_with_relu.set_op("GAP8_Conv2D");
            conv_with_relu.set_name(match.node.name());
            AddNodeInput(conv_node.input(0), &conv_with_relu);
            AddNodeInput(conv_node.input(1), &conv_with_relu);
            AddNodeInput(add_node.input(1), &conv_with_relu);
            CopyNodeAttr(add_node, "T", "T", &conv_with_relu);
            CopyNodeAttr(conv_node, "padding", "padding", &conv_with_relu);
            CopyNodeAttr(conv_node, "strides", "strides", &conv_with_relu);
            CopyNodeAttr(match.node, "padding", "maxpool_padding", &conv_with_relu);
            CopyNodeAttr(match.node, "strides", "maxpool_strides", &conv_with_relu);
            CopyNodeAttr(match.node,"ksize","pooling_factor",&conv_with_relu);
            AddNodeAttr("maxpool",true,&conv_with_relu);
            AddNodeAttr("relu",false,&conv_with_relu);
            CopyNodeAttr(match.node, "_output_shapes", "_output_shapes", &conv_with_relu);
            new_nodes->push_back(conv_with_relu);
            any_nodes_removed = true;
            return Status::OK();
          },
          {}, &replaced_graph_def));
  
        
  *output_graph_def = replaced_graph_def; // need to change graph 
  current_graph_def=replaced_graph_def;
  } while (any_nodes_removed);
  return Status::OK();
}

Status FuseGAP8_Conv2DAndMaxpool( const GraphDef& input_graph_def,
                                const TransformFuncContext& context,
                                GraphDef* output_graph_def) {


  bool any_nodes_removed;
  GraphDef current_graph_def;
  current_graph_def=input_graph_def;
  do {
  any_nodes_removed=false;
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      current_graph_def,  // clang-format of
      {"MaxPool",
        {
              {"GAP8_Conv2D",
                {
                  {"*"},
                  {"*"},
                }
              },  
        }
      },  // clang-format on
      [&any_nodes_removed](
          const NodeMatch& match,
          const std::set<string>& input_nodes,
          const std::set<string>& output_nodes,
          std::vector<NodeDef>* new_nodes) {

            const NodeDef& conv_node = match.inputs[0].inputs[0].node;
            CHECK_EQ("GAP8_Conv2D", conv_node.op());
            const NodeDef& bias_node = match.inputs[0].inputs[2].node;
            const NodeDef& input_node = match.inputs[0].inputs[0].node;
            const NodeDef& weights_node = match.inputs[0].inputs[1].node;
            cerr << "++++++ FuseGAP8_Conv2DAndMaxpool +++++++++" << "\n"; 
            new_nodes->push_back(bias_node);
            new_nodes->push_back(input_node);
            new_nodes->push_back(weights_node);

            NodeDef conv_with_relu;
            conv_with_relu.set_op("GAP8_Conv2D");
            conv_with_relu.set_name(match.node.name());
            AddNodeInput(conv_node.input(0), &conv_with_relu);
            AddNodeInput(conv_node.input(1), &conv_with_relu);
            AddNodeInput(conv_node.input(2), &conv_with_relu);
            CopyNodeAttr(conv_node, "T", "T", &conv_with_relu);
            CopyNodeAttr(conv_node, "padding", "padding", &conv_with_relu);
            CopyNodeAttr(conv_node, "strides", "strides", &conv_with_relu);
            CopyNodeAttr(match.node, "padding", "maxpool_padding", &conv_with_relu);
            CopyNodeAttr(match.node, "strides", "maxpool_strides", &conv_with_relu);
            CopyNodeAttr(match.node,"ksize","pooling_factor",&conv_with_relu);
            CopyNodeAttr(conv_node,"relu","relu",&conv_with_relu);
            AddNodeAttr("maxpool",true,&conv_with_relu);
            CopyNodeAttr(match.node, "_output_shapes", "_output_shapes", &conv_with_relu);
            new_nodes->push_back(conv_with_relu);
            any_nodes_removed=true;

            return Status::OK();
          },
          {}, &replaced_graph_def));
  
        
  *output_graph_def = replaced_graph_def;
  current_graph_def=replaced_graph_def;
  } while (any_nodes_removed);
  return Status::OK();
}


Status FuseConv2DAndAddAndRelu( const GraphDef& input_graph_def,
                                const TransformFuncContext& context,
                                GraphDef* output_graph_def) {

  bool any_nodes_removed;
  GraphDef current_graph_def;
  current_graph_def=input_graph_def;
  do {
  any_nodes_removed=false;
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      current_graph_def,  // clang-format of
      {"Relu",
        {
          {"Add",
            {
              {"Conv2D",
                {
                  {"*"},
                  {"*"},
                }
              },
              {"*"}
            }
          }
        }
      },  // clang-format on
      [&any_nodes_removed](
          const NodeMatch& match,
          const std::set<string>& input_nodes,
          const std::set<string>& output_nodes,
          std::vector<NodeDef>* new_nodes) {

            const NodeDef& add_node = match.inputs[0].node;
            CHECK_EQ("Add", add_node.op());
            const NodeDef& conv_node = match.inputs[0].inputs[0].node;
            CHECK_EQ("Conv2D", conv_node.op());
            const NodeDef& bias_node = match.inputs[0].inputs[1].node;
            const NodeDef& input_node = match.inputs[0].inputs[0].inputs[0].node;
            const NodeDef& weights_node = match.inputs[0].inputs[0].inputs[1].node;
            cerr << "++++++ FuseConv2DAndAddAndRelu +++++++++" << "\n"; 
            new_nodes->push_back(bias_node);
            new_nodes->push_back(input_node);
            new_nodes->push_back(weights_node);

            NodeDef conv_with_relu;
            conv_with_relu.set_op("GAP8_Conv2D");
            conv_with_relu.set_name(match.node.name());
            AddNodeInput(conv_node.input(0), &conv_with_relu);
            AddNodeInput(conv_node.input(1), &conv_with_relu);
            AddNodeInput(add_node.input(1), &conv_with_relu);
            CopyNodeAttr(add_node, "T", "T", &conv_with_relu);
            CopyNodeAttr(conv_node, "padding", "padding", &conv_with_relu);
            CopyNodeAttr(conv_node, "strides", "strides", &conv_with_relu);
            AddNodeAttr("maxpool",false,&conv_with_relu);
            AddNodeAttr("relu",true,&conv_with_relu);
            AddNodeAttr("pooling_factor",0,&conv_with_relu);
            CopyNodeAttr(match.node, "_output_shapes", "_output_shapes", &conv_with_relu);
            new_nodes->push_back(conv_with_relu);
            any_nodes_removed=true;
            return Status::OK();
          },
          {}, &replaced_graph_def));
  
        
  *output_graph_def = replaced_graph_def;
  current_graph_def=replaced_graph_def;
  } while (any_nodes_removed);
  return Status::OK();
}



/*Status FuseConv2DAndMaxpool( const GraphDef& input_graph_def,
                                const TransformFuncContext& context,
                                GraphDef* output_graph_def) {

  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format of
      {"Maxpool",
        {
              {"Conv2D",
                {
                  {"*"},
                  {"*"},
                }
              },
        }
      },  // clang-format on
      [](
          const NodeMatch& match,
          const std::set<string>& input_nodes,
          const std::set<string>& output_nodes,
          std::vector<NodeDef>* new_nodes) {

            const NodeDef& conv_node = match.inputs[0].node;
            CHECK_EQ("Conv2D", conv_node.op());
            const NodeDef& input_node = match.inputs[0].inputs[0].node;
            const NodeDef& weights_node = match.inputs[0].inputs[1].node;

            new_nodes->push_back(input_node);
            new_nodes->push_back(weights_node);
            cerr << "++++++ FuseConv2DAndMaxpool +++++++++" << "\n";  
            NodeDef conv_with_maxpool;
            conv_with_maxpool.set_op("GAP8_Conv2D");
            conv_with_maxpool.set_name(match.node.name());
            AddNodeInput(conv_node.input(0), &conv_with_maxpool);
            AddNodeInput(conv_node.input(1), &conv_with_maxpool);
            CopyNodeAttr(conv_node, "padding", "padding", &conv_with_maxpool);
            CopyNodeAttr(conv_node, "strides", "strides", &conv_with_maxpool);
            CopyNodeAttr(match.node, "padding", "maxpool_padding", &conv_with_maxpool);
            CopyNodeAttr(match.node, "strides", "maxpool_strides", &conv_with_maxpool);
            CopyNodeAttr(match.node,"ksize","ksize",&conv_with_maxpool);
            AddNodeAttr("maxpool",true,&conv_with_maxpool);
            AddNodeAttr("relu",false,&conv_with_maxpool);
            CopyNodeAttr(match.node, "_output_shapes", "_output_shapes", &conv_with_maxpool);
            new_nodes->push_back(conv_with_maxpool);

            return Status::OK();
          },
          {}, &replaced_graph_def));
  
        
  *output_graph_def = replaced_graph_def;
  return Status::OK();
}*/


Status FuseConv2DAndAdd( const GraphDef& input_graph_def,
                                const TransformFuncContext& context,
                                GraphDef* output_graph_def) {
  bool any_nodes_removed;
  GraphDef current_graph_def;
  current_graph_def=input_graph_def;
  do {
  any_nodes_removed=false;
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      current_graph_def,  // clang-format of
      {"Add",
        {
          {"Conv2D",
            {
              {"*"},
              {"*"},
            }
          },
          {"*"}
        }
      },  // clang-format on
      [&any_nodes_removed](
          const NodeMatch& match,
          const std::set<string>& input_nodes,
          const std::set<string>& output_nodes,
          std::vector<NodeDef>* new_nodes) {

            const NodeDef& add_node = match.node;
            CHECK_EQ("Add", add_node.op());
            const NodeDef& conv_node = match.inputs[0].node;
            CHECK_EQ("Conv2D", conv_node.op());
            const NodeDef& bias_node = match.inputs[1].node;
            const NodeDef& input_node = match.inputs[0].inputs[0].node;
            const NodeDef& weights_node = match.inputs[0].inputs[1].node;
            cerr << "++++++ FuseConv2DAndAdd +++++++++" << "\n";
            // We'll be reusing the old weights and bias.
            new_nodes->push_back(bias_node);
            new_nodes->push_back(input_node);
            new_nodes->push_back(weights_node);
            // Set up the new fused version of the convolution op.
            NodeDef conv_with_relu;
            conv_with_relu.set_op("GAP8_Conv2D");
            conv_with_relu.set_name(match.node.name());
            AddNodeInput(conv_node.input(0), &conv_with_relu);
            AddNodeInput(conv_node.input(1), &conv_with_relu);
            AddNodeInput(add_node.input(1), &conv_with_relu);
            CopyNodeAttr(add_node, "T", "T", &conv_with_relu);
            CopyNodeAttr(conv_node, "padding", "padding", &conv_with_relu);
            CopyNodeAttr(conv_node, "strides", "strides", &conv_with_relu);
            AddNodeAttr("maxpool",false,&conv_with_relu);
            AddNodeAttr("relu",false,&conv_with_relu);
            AddNodeAttr("pooling_factor",0,&conv_with_relu);
            CopyNodeAttr(conv_node, "_output_shapes", "_output_shapes", &conv_with_relu);
            new_nodes->push_back(conv_with_relu);
            any_nodes_removed=true;

            return Status::OK();
          },
          {}, &replaced_graph_def));
  
        
  *output_graph_def = replaced_graph_def;
  current_graph_def=replaced_graph_def;
  } while (any_nodes_removed);
  return Status::OK();
}


Status FuseReshapeAndMatmulAndAddAndReluAndSoftmax( 
        const GraphDef& input_graph_def,
        const TransformFuncContext& context,
        GraphDef* output_graph_def) {

  bool any_nodes_removed;
  GraphDef current_graph_def;
  current_graph_def=input_graph_def;
  do {
  any_nodes_removed=false;
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
               current_graph_def,  // clang-format off
               {"Softmax",
                   {
                     {"Relu",
                         {
                            {"Add",
                               {
                                {"MatMul",
                                     {
                                      {"Reshape",
                                           {
                                           {"*"},
                                           {"*"}
                                           }
                                       },
                                      {"*"}
                                      }
                                 },
                                {"*"}
                               }
                            }
                         }
                     }
                   }
                },
               [&any_nodes_removed](
                  const NodeMatch& match,
                  const std::set<string>& input_nodes,
                  const std::set<string>& output_nodes,
                  std::vector<NodeDef>* new_nodes){

                    const NodeDef& softmax_node = match.node;
                    CHECK_EQ("Softmax", softmax_node.op());
                    const NodeDef& relu_node = match.inputs[0].node;
                    CHECK_EQ("Relu", relu_node.op());
                    const NodeDef& add_node = match.inputs[0].inputs[0].node;
                    CHECK_EQ("Add", add_node.op());
                    const NodeDef& matmul_node = match.inputs[0].inputs[0].inputs[0].node;
                    CHECK_EQ("MatMul", matmul_node.op());
                    const NodeDef& reshape_node = match.inputs[0].inputs[0].inputs[0].inputs[0].node;
                    CHECK_EQ("Reshape", reshape_node.op());

                    const NodeDef& bias_node = match.inputs[0].inputs[0].inputs[1].node;
                    const NodeDef& weights_node = match.inputs[0].inputs[0].inputs[0].inputs[1].node;
                    const NodeDef& shape_node = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[1].node;
                    const NodeDef& input_node = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].node;
                    cerr << "++++++ FuseReshapeAndMatmulAndAddAndReluAndSoftmax +++++++++" << "\n";
                    new_nodes->push_back(bias_node);
                    new_nodes->push_back(weights_node);
                    new_nodes->push_back(input_node);
                    new_nodes->push_back(shape_node);


                    NodeDef dense_node;
                    dense_node.set_op("GAP8_DenseLayer");
                    dense_node.set_name(match.node.name());
                    AddNodeInput(reshape_node.input(0), &dense_node);
                    AddNodeInput(reshape_node.input(1), &dense_node);
                    AddNodeInput(add_node.input(1), &dense_node);
                    AddNodeInput(matmul_node.input(1), &dense_node);
              //      AddNodeInput(matmul_node.input(1), &dense_node);
                    CopyNodeAttr(softmax_node, "T", "T", &dense_node);
                    CopyNodeAttr(softmax_node, "_output_shapes", "_output_shapes", &dense_node);
                    AddNodeAttr("relu",true,&dense_node);
                    new_nodes->push_back(dense_node);
                    any_nodes_removed=true;

                      return Status::OK();

                  },
                  {}, &replaced_graph_def));
  *output_graph_def = replaced_graph_def;
  current_graph_def=replaced_graph_def;
  } while (any_nodes_removed);
  return Status::OK();
}


Status FuseReshapeAndMatmulAndAddAndSoftmax( 
        const GraphDef& input_graph_def,
        const TransformFuncContext& context,
        GraphDef* output_graph_def) {

  bool any_nodes_removed;
  GraphDef current_graph_def;
  current_graph_def=input_graph_def;
  do {
  any_nodes_removed=false;
 GraphDef replaced_graph_def;
 TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
          current_graph_def,  // clang-format off
                {"Softmax",
                    {
                       {"Add",
                          {
                           {"MatMul",
                              {
                                 {"Reshape",
                                    {
                                      {"*"},
                                      {"*"}
                                    }
                                  },
                                 {"*"}
                               }
                            },
                           {"*"}
                          }
                       }
                   }
                },
          [&any_nodes_removed]( 
              const NodeMatch& match,
             const std::set<string>& input_nodes,
             const std::set<string>& output_nodes,
             std::vector<NodeDef>* new_nodes){

               const NodeDef& softmax_node = match.node;
               CHECK_EQ("Softmax", softmax_node.op());
               const NodeDef& add_node = match.inputs[0].node;
               CHECK_EQ("Add", add_node.op());
               const NodeDef& matmul_node = match.inputs[0].inputs[0].node;
               CHECK_EQ("MatMul", matmul_node.op());
               const NodeDef& reshape_node = match.inputs[0].inputs[0].inputs[0].node;
               CHECK_EQ("Reshape", reshape_node.op());

               const NodeDef& bias_node = match.inputs[0].inputs[1].node;
               const NodeDef& weights_node = match.inputs[0].inputs[0].inputs[1].node;
               const NodeDef& shape_node = match.inputs[0].inputs[0].inputs[0].inputs[1].node;
               const NodeDef& input_node = match.inputs[0].inputs[0].inputs[0].inputs[0].node;
               cerr << "++++++ FuseReshapeAndMatmulAndAddAndSoftmax +++++++++" << "\n";
               new_nodes->push_back(bias_node);
               new_nodes->push_back(weights_node);
               new_nodes->push_back(input_node);
               new_nodes->push_back(shape_node);


               NodeDef dense_node;
               dense_node.set_op("GAP8_DenseLayer");
               dense_node.set_name(match.node.name());
               AddNodeInput(reshape_node.input(0), &dense_node);
               AddNodeInput(reshape_node.input(1), &dense_node);
               AddNodeInput(add_node.input(1), &dense_node);
                 AddNodeInput(matmul_node.input(1), &dense_node);
            //   AddNodeInput(matmul_node.input(1), &dense_node);
               CopyNodeAttr(softmax_node, "T", "T", &dense_node);
               CopyNodeAttr(softmax_node, "_output_shapes", "_output_shapes", &dense_node);
               AddNodeAttr("relu",false,&dense_node);
               new_nodes->push_back(dense_node);
               any_nodes_removed=true;

                 return Status::OK();

             },
             {}, &replaced_graph_def));

 *output_graph_def = replaced_graph_def;
 current_graph_def=replaced_graph_def;
  } while (any_nodes_removed);
  return Status::OK();
}


  Status FuseReshapeAndMatmulAndAdd(
					      const GraphDef& input_graph_def,
					      const TransformFuncContext& context,
					      GraphDef* output_graph_def) {
    bool any_nodes_removed;
    GraphDef current_graph_def;
    current_graph_def=input_graph_def;
    do {
      any_nodes_removed=false;
      GraphDef replaced_graph_def;
      TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
						current_graph_def,  // clang-format off
						{"Add",
							  {
							    {"MatMul",
								{
								  {"Reshape",
								      {
									{"*"},
									  {"*"}
								      }
								  },
								    {"*"}
								}
							    },
							      {"*"}
							  }
						},
						[&any_nodes_removed](
								     const NodeMatch& match,
								     const std::set<string>& input_nodes,
								     const std::set<string>& output_nodes,
								     std::vector<NodeDef>* new_nodes){

						  const NodeDef& add_node = match.node;
						  CHECK_EQ("Add", add_node.op());
						  const NodeDef& matmul_node = match.inputs[0].node;
						  CHECK_EQ("MatMul", matmul_node.op());
						  const NodeDef& reshape_node = match.inputs[0].inputs[0].node;
						  CHECK_EQ("Reshape", reshape_node.op());

						  const NodeDef& bias_node = match.inputs[1].node;
						  const NodeDef& weights_node = match.inputs[0].inputs[1].node;
						  const NodeDef& shape_node = match.inputs[0].inputs[0].inputs[1].node;
						  const NodeDef& input_node = match.inputs[0].inputs[0].inputs[0].node;
						  cerr << "++++++ FuseReshapeAndMatmulAndAdd +++++++++" << "\n";
						  new_nodes->push_back(bias_node);
						  new_nodes->push_back(weights_node);
						  new_nodes->push_back(input_node);
						  new_nodes->push_back(shape_node);

						  NodeDef dense_node;
						  dense_node.set_op("GAP8_DenseLayer");
						  dense_node.set_name(match.node.name());
						  AddNodeInput(reshape_node.input(0), &dense_node);
						  AddNodeInput(reshape_node.input(1), &dense_node);
						  AddNodeInput(add_node.input(1), &dense_node);
						  AddNodeInput(matmul_node.input(1), &dense_node);
						  //   AddNodeInput(matmul_node.input(1), &dense_node);
						  CopyNodeAttr(add_node, "T", "T", &dense_node);
						  CopyNodeAttr(add_node, "_output_shapes", "_output_shapes", &dense_node);
						  AddNodeAttr("relu",false,&dense_node);
						  new_nodes->push_back(dense_node);
						  any_nodes_removed=true;

						  return Status::OK();

						},
						{}, &replaced_graph_def));

      *output_graph_def = replaced_graph_def;
      current_graph_def=replaced_graph_def;
    } while (any_nodes_removed);
    return Status::OK();
  }
   
REGISTER_GRAPH_TRANSFORM("fuse_conv2d_add_relu_maxpool", FuseConv2DAndAddAndReluAndMaxpool);

REGISTER_GRAPH_TRANSFORM("fuse_conv2d_add_relu", FuseConv2DAndAddAndRelu);

REGISTER_GRAPH_TRANSFORM("fuse_conv2d_add_maxpool", FuseConv2DAndAddAndMaxpool);

REGISTER_GRAPH_TRANSFORM("fuse_conv2d_add", FuseConv2DAndAdd);

REGISTER_GRAPH_TRANSFORM("fuse_GAP8_conv2d_maxpool",FuseGAP8_Conv2DAndMaxpool);

REGISTER_GRAPH_TRANSFORM("fuse_reshape_matmul_add_relu_softmax",FuseReshapeAndMatmulAndAddAndReluAndSoftmax);

REGISTER_GRAPH_TRANSFORM("fuse_reshape_matmul_add_softmax", FuseReshapeAndMatmulAndAddAndSoftmax);

REGISTER_GRAPH_TRANSFORM("fuse_reshape_matmul_add", FuseReshapeAndMatmulAndAdd);
  
} // namespace graph_transforms
} // namespace tensorflow
