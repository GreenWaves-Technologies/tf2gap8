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
Contributors: Nicolas Lepetit
*/


#include "DenseLayer.h"

using namespace std;


DenseLayer::DenseLayer(std::string name,int i , int n_in, int n_out, int h_o, 
                        int w_o, int h, int w, int act, bool relu, int norm_factor, 
                        int norm_bias ) :
    Layer::Layer(name, "dense",i , n_in, h, w), n_out_feat(n_out), 
                output_height(h_o), output_width(w_o), activation(act), 
                relu(relu), norm_factor(norm_factor), norm_bias(norm_bias){}

DenseLayer::DenseLayer(std::string name,int i , int n_in, int n_out, int h_o, 
                        int w_o, int h, int w, int act) :
    Layer::Layer(name, "dense",i , n_in, h, w), n_out_feat(n_out), 
                output_height(h_o), output_width(w_o), activation(act) {}

std::string DenseLayer::Summary(){

        std::string ret ="This is a Denser Layer called: " + this->name + "\n" ;
        ret = ret + "It has the following parameters: \n";
        ret = ret + "   -n_input:  "+ std::to_string(this->n_in_feat) + "\n";
        ret = ret + "   -n_output: "+ std::to_string(this->n_out_feat) + "\n";
        ret = ret + "   -output_height:  "+ std::to_string(this->output_height) + "\n";
        ret = ret + "   -output_width:  "+ std::to_string(this->output_width) + "\n";
        ret = ret + "   -intput_height:  "+ std::to_string(this->input_height) + "\n";
        ret = ret + "   -intput_width:  "+ std::to_string(this->input_width) + "\n";
        ret = ret + "   -relu: " + std::to_string(this->relu) + "\n";
        ret = ret + "   -norm_factor: " + std::to_string(this->norm_factor) + "\n";
        ret = ret + "   -norm_bias: " + std::to_string(this->norm_bias) + "\n";
        if (this->activation == 1){

            ret = ret + "   -with RELU"+ "\n";
        }
        else{
            ret = ret + "   -without RELU"+ "\n";
        }
        ret = ret + "\n";
        return ret;
}
