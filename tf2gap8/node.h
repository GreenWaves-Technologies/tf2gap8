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

/*
Contributors: Nicolas Lepetit, Corine Lamagdeleine, Joel Cambonie
*/
#ifndef NODE_H_
#define NODE_H_

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include <string>

std::string get_node_name(const tensorflow::NodeDef &node);
std::string get_node_operation(const tensorflow::NodeDef &node);
std::string get_node_type(const tensorflow::NodeDef &node);
std::string get_node_dtype(const tensorflow::NodeDef &node);
std::string get_node_output_shapes(const tensorflow::NodeDef &node);
std::string get_conv_padding(const tensorflow::NodeDef &node);
std::string get_conv_strides(const tensorflow::NodeDef &node);
std::string get_maxpool_padding(const tensorflow::NodeDef &node);
std::string get_maxpool_strides(const tensorflow::NodeDef &node);
std::string get_maxpool_ksize(const tensorflow::NodeDef &node);
std::string get_GAP8_conv2d_pooling(const tensorflow::NodeDef &node);
std::string get_node_input(const tensorflow::NodeDef &node);
int get_pooling_factor(const tensorflow::NodeDef &node);
int get_pooling_factor_conv2d(const tensorflow::NodeDef &node);
int get_output_number(const tensorflow::NodeDef &node);
int get_output_height(const tensorflow::NodeDef &node);
int get_output_width(const tensorflow::NodeDef &node);
void settingOutputVariable(const tensorflow::NodeDef &node,
			    int& n_out_feat,
			    int& output_height,
			    int& output_width);
void settingDenseOutputVariable(const tensorflow::NodeDef &node,
        int& n_out_feat,
        int& output_height,
        int& output_width);
#endif
