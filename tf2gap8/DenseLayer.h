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
This is the Class of a Dense Layer, that will be usefull to construct the C program
Contributor: Nicolas Lepetit, Corine Lamagdeleine
*/

#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "Layer.h"

using namespace std;

class DenseLayer : public Layer {
 public:
  int n_out_feat; // number of output features
  int activation; // activation function
  int output_height; // output height
  int output_width; // output width
  bool relu; // True if Dense layer has a RElu operator
  int norm_factor; //normalization factor
  int norm_bias; // normalization bias

  //constructors
  DenseLayer(std::string name, int i, int n_in, int n_out, 
	     int h_o, int h_w, int h, int w, int act, bool relu, int norm_factor, 
	     int norm_bias);
  DenseLayer(std::string name, int i, int n_in, int n_out, 
	     int h_o, int h_w, int h, int w, int act);
  std::string Summary();

};

#endif
