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
This is the Class of a Conv Layer, that will be usefull to construct the C program
Contributor: Nicolas LePetit, Corine Lamagdeleine
*/

#ifndef DEPTHWISE_CONV_LAYER_H
#define DEPTHWISE_CONV_LAYER_H

#include "Layer.h"

using namespace std;

// Class to store a convolutional layer before the
// GAP8 Code generation
class DepthwiseConvLayer : public Layer {
 public:
  int n_out_feat; // number of output features 
  int filter_width; // convolutional filter width
  int filter_height; // convolutional filter height
  bool relu; // True if covolutional layer has a Relu operator
  bool maxpool; // True if convolutional layer has a maxpool operator
  int pooling_factor; // pooling factor if maxpool true
  int norm_factor; // normalization factor

  // Constructors
  DepthwiseConvLayer(std::string name, int i, int n_in, int n_out, int fil_width, int fil_height, 
	    int h, int w);
  DepthwiseConvLayer(std::string name, int i, int n_in, int n_out, int fil_width, int fil_height, 
	    bool relu, bool maxpool, int pooling_factor, int h, int w, int norm_factor);
  
  // prints a summary of a class object
  std::string Summary();

};

#endif
