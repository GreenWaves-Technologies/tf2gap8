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

namespace tensorflow {
namespace graph_transforms {


Status Gap8TransformTool( const GraphDef& input_graph_def,
                          const TransformFuncContext& context,
                          GraphDef* output_graph_def) {

      GraphDef inter_graph = input_graph_def;
      int NodeNumber = 0;

      for (const NodeDef& node: input_graph_def.node() ){
          NodeNumber += 1;
      }

      for (int j = 0; j < NodeNumber; j++){
          GraphDef temp_graph_1;
          TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
          inter_graph,  // clang-format of
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
      [](const NodeMatch& match,
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

            new_nodes->push_back(bias_node);
            new_nodes->push_back(input_node);
            new_nodes->push_back(weights_node);

            NodeDef conv_with_relu;
            conv_with_relu.set_op("GAP8_ConvLayer_ReLu");
            conv_with_relu.set_name(match.node.name());
            AddNodeInput(conv_node.input(0), &conv_with_relu);
            AddNodeInput(conv_node.input(1), &conv_with_relu);
            AddNodeInput(add_node.input(1), &conv_with_relu);
            CopyNodeAttr(add_node, "T", "T", &conv_with_relu);
            CopyNodeAttr(conv_node, "padding", "padding", &conv_with_relu);
            CopyNodeAttr(conv_node, "strides", "strides", &conv_with_relu);
            CopyNodeAttr(conv_node, "_output_shapes", "_output_shapes", &conv_with_relu);
            new_nodes->push_back(conv_with_relu);

            return Status::OK();
          },
          {}, &temp_graph_1));



          GraphDef temp_graph_2;
          TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
          temp_graph_1,  // clang-format off
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
        [](const NodeMatch& match,
           const std::set<string>& input_nodes,
           const std::set<string>& output_nodes,
           std::vector<NodeDef>* new_nodes) {
          // Find all the nodes we expect in the subgraph.

          const NodeDef& add_node = match.node;
          CHECK_EQ("Add", add_node.op());
          const NodeDef& conv_node = match.inputs[0].node;
          CHECK_EQ("Conv2D", conv_node.op());
          const NodeDef& bias_node = match.inputs[1].node;
          const NodeDef& input_node = match.inputs[0].inputs[0].node;
          const NodeDef& weights_node = match.inputs[0].inputs[1].node;

          // We'll be reusing the old weights and bias.
          new_nodes->push_back(bias_node);
          new_nodes->push_back(input_node);
          new_nodes->push_back(weights_node);

            // Set up the new fused version of the convolution op.
          NodeDef conv_with_relu;
          conv_with_relu.set_op("GAP8_ConvLayer");
          conv_with_relu.set_name(match.node.name());
          AddNodeInput(conv_node.input(0), &conv_with_relu);
          AddNodeInput(conv_node.input(1), &conv_with_relu);
          AddNodeInput(add_node.input(1), &conv_with_relu);
          CopyNodeAttr(add_node, "T", "T", &conv_with_relu);
          CopyNodeAttr(conv_node, "padding", "padding", &conv_with_relu);
          CopyNodeAttr(conv_node, "strides", "strides", &conv_with_relu);
          CopyNodeAttr(conv_node, "_output_shapes", "_output_shapes", &conv_with_relu);
          new_nodes->push_back(conv_with_relu);

          return Status::OK();
        },
        {}, &temp_graph_2));

        GraphDef temp_graph_3;

        TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
          temp_graph_2,  // clang-format off
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
          [](const NodeMatch& match,
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
               new_nodes->push_back(dense_node);

                 return Status::OK();

             },
             {}, &temp_graph_3));

             GraphDef temp_graph_4;

             TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
               temp_graph_3,  // clang-format off
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
               [](const NodeMatch& match,
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

                    new_nodes->push_back(bias_node);
                    new_nodes->push_back(weights_node);
                    new_nodes->push_back(input_node);
                    new_nodes->push_back(shape_node);


                    NodeDef dense_node;
                    dense_node.set_op("GAP8_DenseLayer_ReLu");
                    dense_node.set_name(match.node.name());
                    AddNodeInput(reshape_node.input(0), &dense_node);
                    AddNodeInput(reshape_node.input(1), &dense_node);
                    AddNodeInput(add_node.input(1), &dense_node);
                    AddNodeInput(matmul_node.input(1), &dense_node);
              //      AddNodeInput(matmul_node.input(1), &dense_node);
                    CopyNodeAttr(softmax_node, "T", "T", &dense_node);
                    CopyNodeAttr(softmax_node, "_output_shapes", "_output_shapes", &dense_node);
                    new_nodes->push_back(dense_node);


                      return Status::OK();

                  },
                  {}, &temp_graph_4));

                  GraphDef temp_graph_5;


          TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
            temp_graph_4,
            {"Relu",
                {
                    {"GAP8_ConvLayer",
                          {
                                {"*"},
                                {"*"},
                                {"*"},
                          }
                    }
                }
            },
            [](const NodeMatch& match,
               const std::set<string>& input_nodes,
               const std::set<string>& output_nodes,
               std::vector<NodeDef>* new_nodes){
                  const NodeDef& relu_node = match.node;
                  CHECK_EQ("Relu", relu_node.op());
                  const NodeDef& GAP8_ConvLayer = match.inputs[0].node;
                  CHECK_EQ("GAP8_ConvLayer", GAP8_ConvLayer.op());
                  const NodeDef& input_node = match.inputs[0].inputs[0].node;
                  const NodeDef& weights_node = match.inputs[0].inputs[1].node;
                  const NodeDef& bias_node = match.inputs[0].inputs[2].node;

                  new_nodes->push_back(bias_node);
                  new_nodes->push_back(input_node);
                  new_nodes->push_back(weights_node);

                    // Set up the new fused version of the convolution op.
                  NodeDef GAP8_ConvLayer_ReLu;
                  GAP8_ConvLayer_ReLu.set_op("GAP8_ConvLayer_ReLu");
                  GAP8_ConvLayer_ReLu.set_name(match.node.name());
                  AddNodeInput(GAP8_ConvLayer.input(0), &GAP8_ConvLayer_ReLu);
                  AddNodeInput(GAP8_ConvLayer.input(1), &GAP8_ConvLayer_ReLu);
                  AddNodeInput(GAP8_ConvLayer.input(2), &GAP8_ConvLayer_ReLu);
                  CopyNodeAttr(relu_node, "T", "T", &GAP8_ConvLayer_ReLu);
                  CopyNodeAttr(GAP8_ConvLayer, "padding", "padding", &GAP8_ConvLayer_ReLu);
                  CopyNodeAttr(GAP8_ConvLayer, "strides", "strides", &GAP8_ConvLayer_ReLu);
                  CopyNodeAttr(relu_node, "_output_shapes", "_output_shapes", &GAP8_ConvLayer_ReLu);
                  new_nodes->push_back(GAP8_ConvLayer_ReLu);

                  return Status::OK();

               },
          {}, &temp_graph_5));


                  inter_graph = temp_graph_5;
                }

      *output_graph_def = inter_graph;



    return Status::OK();
  }

REGISTER_GRAPH_TRANSFORM("gap8_transform_tool", Gap8TransformTool);

}
}
