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
*/

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

Status FuseConv2DAndAddAndReluAndMaxpool( const GraphDef& input_graph_def,
                                          const TransformFuncContext& context,
                                          GraphDef* output_graph_def);

Status FuseConv2DAndAddAndRelu(	const GraphDef& input_graph_def,
                            	const TransformFuncContext& context,
                            	GraphDef* output_graph_def);

Status FuseConv2DAndAddAndMaxpool( const GraphDef& input_graph_def,
                        		   const TransformFuncContext& context,
                        		   GraphDef* output_graph_def);
Status FuseConv2DAndAdd( const GraphDef& input_graph_def,
                                const TransformFuncContext& context,
                                GraphDef* output_graph_def);
Status FuseGAP8_Conv2DAndMaxpool( const GraphDef& input_graph_def,
                                const TransformFuncContext& context,
                                GraphDef* output_graph_def);

Status FuseReshapeAndMatmulAndAddAndReluAndSoftmax( 
        const GraphDef& input_graph_def,
        const TransformFuncContext& context,
        GraphDef* output_graph_def);

Status FuseReshapeAndMatmulAndAddAndSoftmax( 
        const GraphDef& input_graph_def,
        const TransformFuncContext& context,
        GraphDef* output_graph_def);

Status FuseReshapeAndMatmulAndAdd(
	const GraphDef& input_graph_def,
	const TransformFuncContext& context,
        GraphDef* output_graph_def);

Status FuseReshapeAndMatmulAndAddAndRelu(
	const GraphDef& input_graph_def,
	const TransformFuncContext& context,
        GraphDef* output_graph_def);

Status FuseMatmulAndAddAndRelu(
	const GraphDef& input_graph_def,
	const TransformFuncContext& context,
        GraphDef* output_graph_def);
Status FuseMatmulAndAdd(
	const GraphDef& input_graph_def,
	const TransformFuncContext& context,
        GraphDef* output_graph_def);

}
}
