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
*    documentation and/or other materials provided with tcouthe distribution.
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
This is the main module of the Tensorflow to GAP8 bridge. It takes an optimized graph
(obtained through a freeze_graph of a Tensorflow application graph and training 
checkpoint, followed by a graph!transformation procedure) and generates the 
corresponding source code to work with the GreenWaves Technologies GAP8 IoT processor.
contributors: Nicolas Lepetit, Corine Lamagdeleine, Joel Cambonie
This version is prepared for the April2 2018 release

*/

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/attr_value.pb_text.h"
#include "tensorflow/core/framework/tensor.pb_text.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb_text.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "node.h"
#include "Layer.h"
#include "ConvLayer.h"
#include "PoolLayer.h"
#include "DenseLayer.h"
#include "GAP8Tensor.h"
#include "tf2gap8-exception.h"
#include "colormod.h"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <algorithm>

// define fixed point arithmetic 16-QF,QF
#define QF 14
// no more used
//int size_input = 0;

using namespace tensorflow;
using namespace std;

enum type_layer  {conv,dense,placeholder,argmax};

typedef struct  {
  std::string name="";

  std::string input="";
  type_layer type  ;
  char activation_pool=0;
  char activation_relu=0;
  int n_in_feat; // number of features in 
  int n_out_feat; // number of features out
  char filter_width;
  char filter_height;
  int input_height,input_width;
  int out_width,out_height;
  char pooling_factor;


} layer_t;

#define MAX_LAYER_NB 100 // maximum number of layers we can process


char still_alive(layer_t *l_layer, int num, int index) {

  string namin;
  
  for(int i=index+1;i<MAX_LAYER_NB;i++) {
    if (!l_layer[i].name.compare("")) return 0;
    namin = l_layer[i].input.substr(0, l_layer[i].input.find("/"));
    if (!l_layer[num].name.compare(namin)) return 1; 
  }

  return 0;
}

// searches the layer connected to the input of layer "num"
char get_input_layer_nb(layer_t *l_layer, int num) {
  char input_layer_nb=-1;
  string namin = l_layer[num].input.substr(0, l_layer[num].input.find("/"));
  clog << " Input ==> " << namin.c_str() << "\n";
  for(int i=0;i<num;i++) {
    if (!l_layer[i].name.compare("")) return -1;
    if (!l_layer[i].name.compare(namin)) return i; 
  }

  return input_layer_nb;
}


// fills a conv layer parameter fields
void fill_layer_param_conv(layer_t *l_layer, int lnum,int n_out_f,int out_w,int out_h, 
                          char filter_width, char filter_height, char act_relu) {
  layer_t *lc = l_layer+lnum;

  lc->n_out_feat = n_out_f;
  lc->out_width = out_w;
  lc->out_height = out_h;
  lc->filter_width = filter_width;
  lc->filter_height = filter_height;
  lc->activation_relu = act_relu;

}

// Fills a pool layer parameter fields
void fill_layer_param_pool(layer_t *l_layer, int lnum,char pool_fact) {
  layer_t *lc = l_layer+lnum;

  lc->activation_pool = 1;
  lc->pooling_factor = pool_fact;
  lc->out_width = lc->out_width/pool_fact;
  lc->out_height = lc->out_height/pool_fact;


}

// Fills a dense layer parameter fields
void fill_layer_param_dense(layer_t *l_layer, int lnum,int n_out_f, char act_relu) {
  layer_t *lc = l_layer + lnum;
  lc->n_out_feat = n_out_f;
  lc->activation_relu = act_relu;
  // Will be always 1 ..
  lc->out_width = 1;
  lc->out_height = 1;

}

// transforms a string to its "ready to be printed" corresponding string with quotes. 
string toGAP8String(const string& str) {
  return strings::StrCat("\"", str_util::CEscape(str), "\"");
}

// Transforms a Tensorflow Tensor into a string corresponding to its GAP8 equivalent
// The ftp boolean indicates if bias or weights parameters have to be generated in floating point
// value or not
string toGAP8Tensor(const TensorProto& tensor_proto, int last_out, int last_w, int last_h,
		    int& filter_height, int& filter_width, bool ftp) {
  Tensor t;

  if (!t.FromProto(tensor_proto)) {
    tf2gap8Exception exc1(strings::StrCat("Error #tg1 in function toGAP8Tensor: <Invalid TensorProto: ",
                           ProtoShortDebugString(tensor_proto), ">. Please contact GWT support."));
    throw exc1;
  }
  // Get the filter height and width in case this tensor represents the Conv2D filter
  filter_height=t.shape().dim_size(0);
  filter_width=t.shape().dim_size(1);
  
  // Here, we need to get the dimensions of the tensor (1D, 2D, 3D , etc..?)
  string ret ="";
  int dims=t.shape().dims();
  strings::StrAppend(&ret,"[]");
  strings::StrAppend(&ret, "=");
  GAP8Tensor GAP8t(t);
  if (dims>0)
    strings::StrAppend(&ret, "{\n");

  strings::StrAppend(&ret, GAP8t.GAP8SummarizeValue(GAP8t.NumElements(), 
						    last_out, last_w, last_h,ftp));
  if (dims>0)
      strings::StrAppend(&ret, "}");

  strings::StrAppend(&ret, ";\n");
  return ret;
}

// Transforms a GAP8 Attribute value into a string corresponding to its 
// GAP8 equivalent
// The ftp boolean indicated if we want the weight and bias parameters generation in floating point or not
string toGAP8AttrValue(const AttrValue& attr_value,
		       int last_out,
		       int last_w,
		       int last_h,
		       int& filter_height,
		       int& filter_width,
		       bool ftp) {
  switch (attr_value.value_case()) {
    case AttrValue::kS:
      return toGAP8String(attr_value.s());
    case AttrValue::kI:
      return strings::StrCat(attr_value.i());
    case AttrValue::kF:
      return strings::StrCat(attr_value.f());
    case AttrValue::kB:
      return attr_value.b() ? "true" : "false";
    case AttrValue::kType:
      return EnumName_DataType(attr_value.type());
    case AttrValue::kShape:
      return PartialTensorShape::DebugString(attr_value.shape());
    case AttrValue::kTensor:
      return toGAP8Tensor(attr_value.tensor(), last_out, last_w, last_h,filter_height, filter_width,ftp);
    case AttrValue::kList: {
      string ret = "[";
      if (attr_value.list().s_size() > 0) {
        for (int i = 0; i < attr_value.list().s_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret, toGAP8String(attr_value.list().s(i)));
        }
      } else if (attr_value.list().i_size() > 0) {
        for (int i = 0; i < attr_value.list().i_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret, attr_value.list().i(i));
        }
      } else if (attr_value.list().f_size() > 0) {
        for (int i = 0; i < attr_value.list().f_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret, attr_value.list().f(i));
        }
      } else if (attr_value.list().b_size() > 0) {
        for (int i = 0; i < attr_value.list().b_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret, attr_value.list().b(i) ? "true" : "false");
        }
      } else if (attr_value.list().type_size() > 0) {
        for (int i = 0; i < attr_value.list().type_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret,
                             EnumName_DataType(attr_value.list().type(i)));
        }
      } else if (attr_value.list().shape_size() > 0) {
        for (int i = 0; i < attr_value.list().shape_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(
              &ret, TensorShape::DebugString(attr_value.list().shape(i)));
        }
      } else if (attr_value.list().tensor_size() > 0) {
        for (int i = 0; i < attr_value.list().tensor_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret,
                             toGAP8Tensor(attr_value.list().tensor(i), 
					  last_out, last_w, last_h,filter_height, filter_width,ftp));
        }
      }

      strings::StrAppend(&ret, "]");
      return ret;
    }
    case AttrValue::kFunc: {
      std::vector<string> entries;
      for (auto p : attr_value.func().attr()) {
        entries.push_back(
            strings::StrCat(p.first, "=", SummarizeAttrValue(p.second)));
      }
      sort(entries.begin(), entries.end());
      return strings::StrCat(attr_value.func().name(), "[",
                             str_util::Join(entries, ", "), "]");
    }
    case AttrValue::kPlaceholder:
      return strings::StrCat("$", attr_value.placeholder());
    case AttrValue::VALUE_NOT_SET:
      {
      tf2gap8Exception exc2(strings::StrCat("Error #tg2 in function toGAP8AttrValue: Attribute Value not set. Please contact GWT support."));
      throw exc2;
      }
  }
  return "";  // Prevents missing return string
}



string toGAP8NodeDef(const NodeDef& node_def, int layer_number, char k, 
		     int last_out, int last_w, int last_h,
		     int& filter_height,
		     int& filter_width,
		     bool ftp){
  // Write to C file weight or Bias Table
  // Here, nodes have been filtered and are only nodes which operator is "Const"
  // For conv2d operators, these represent the nodes were weight and bias formats
  // are stored. So we need to get their value. 
  // We will care only of the attributes "dtype" and "value" . If they do not
  // exist, we will output an error message
  // Please read Weight Formats doc : https://www.tensorflow.org/extend/tool_developers/ 

  // Here we will cast the weight values into short integers
  string ret = strings::StrCat("RT_L2_DATA short int", " ");
  // We get and sort the attributes of the node.
  std::vector<string> attr_names;
  attr_names.reserve(node_def.attr().size());
  // get all attribute names  of the node
  for (const auto& attr : node_def.attr()) {
    attr_names.push_back(attr.first);
  }
  // sort all attribute names  alphabetically
  std::sort(attr_names.begin(), attr_names.end());
  
  // Prepare GAP8 code to generate depending of the nature of this attribute
  if (k==1){
  	strings::StrAppend(&ret, " L2_W_", std::to_string(layer_number));
  }
  else if (k==0) {
  	strings::StrAppend(&ret, " L2_B_", std::to_string(layer_number));
  }
  else {
  	strings::StrAppend(&ret, " L2_X");
  }
  
  // Now get the "value" attribute and transform it in GAP8 formats
  tf2gap8Exception exc3(string("Error #tg3 in function toGAP8ANodeDef: Attribute 'value' not found for node. Please contact GWT support.")); 
  if (k==0 || k==1) {
    //if (tensorflHasNodeAttr(node_def, "value")) {
	 auto iter2 = node_def.attr().find("value");
	 strings::StrAppend(&ret, toGAP8AttrValue(iter2->second, last_out, last_w, last_h, filter_height, filter_width,ftp));
  }
  else {}
  return ret;
}

// Finds a node of a GraphDef representation by its name
const NodeDef* FindNodebyName(const string& name, const GraphDef& graph){
    for (const NodeDef& node: graph.node() ){
        std::string node_name = get_node_name(node);
        if(!node_name.compare(name)){
              return &node;
        }
    }
  return nullptr;
}

// Checks if a directory exists
Status dirExists(std::string dirName){
    DIR* dir = opendir(dirName.c_str());
    if (dir)
    {
        /* Directory exists. */
        closedir(dir);
        return Status();
    }
    else if (ENOENT == errno)
    {
        return errors::NotFound(string("Directory") + dirName + "does not exist");
    }
    else
    {
        return errors::Unknown(string("opendir() failed on directory") + dirName);
    }
}

// Creates a directory
void createDir(string dirName){
    // Creates a directory if it dosn't already exist
    std::string command;
    Status status;
    status=dirExists(dirName);
    if (!status.ok()) {
        // create the directory
        command= string("mkdir ") + dirName;
        system(command.c_str());
    }
}

// Fills a convolutional layer weight template
void fillConvWeightTemplate(std::string& l2_data,
                            int i){

  l2_data = l2_data + "#include \"./inc/L2/L2_W_" +std::to_string(i)+".h\" \n";
}

// Fills a Dense Layer Weight Template
void fillDenseWeightTemplate(std::string& initWandB,
                            std::string& l2_data,
                            int i,
                            int lastOutputWidth,
                            int lastOutputHeight,
                            int OutputNumber,
                            int lastOutputNumber){

    l2_data = l2_data  + "#include \"./inc/L2/L2_W_" +std::to_string(i)+".h\" \n";
}

// Fills header file of name "name"
// The ftp boolean indicated of we want the generation of weights and bias in floating point or not
void fillHeaderFile(ofstream& HFile,
                    ofstream& CFile, 
                    const NodeDef* test,
                    int i,
                    char W,
                    int lastOutputNumber,
                    int lastOutputWidth,
                    int lastOutputHeight,
		    int& filter_height,
		    int& filter_width,
		    bool ftp){
  // Write to C file weight or Bias Table
  string ret= toGAP8NodeDef(*test, i, W,lastOutputNumber, lastOutputWidth,lastOutputHeight,filter_height,filter_width,ftp);
  CFile << ret;
  // Write to  H File corresponding table definition
  ret.erase(0,11);
  std::size_t found = ret.find("=");
  if (found!=std::string::npos)
    ret.erase(found);
  ret.insert(0,"extern ");
  ret.append(";\n");
  HFile << ret;
}

// Creates all the hierarchy of the GAP8Code Directory
void createGAP8CodeDir(string GAP8CodeDirName){
    // Create all necessary directories within the graphDir for GAP8 
    // code generation
    createDir(GAP8CodeDirName);
}

// Setting the output parameters of a NodeDef
/*void settingOutputVariable(const NodeDef& node,
                           int& OutputNumber,
                           int& OutputHeight,
                           int& OutputWidth){

      OutputWidth = get_output_width(node);
      OutputHeight = get_output_height(node);
      OutputNumber = get_output_number(node);

}
*/

// Setting Last Output variable parameters
void settingLastOutputVariable(int OutputNumber,
                               int OutputHeight,
                               int OutputWidth,
                               int& lastOutputNumber,
                               int& lastOutputHeight,
                               int& lastOutputWidth){

      lastOutputWidth = OutputWidth;
      lastOutputHeight = OutputHeight;
      lastOutputNumber = OutputNumber;

}


// Gets the layer name of a node
std::string get_layer_name(const NodeDef& node) {

  std::string ret;
  std::string name;
  name=get_node_name(node);
  ret  = name.substr(0, name.find("/"));
  return ret;
}

// Gets the layer number of a node
int get_layer_num(layer_t *l_layer, std::string name) {
  for (int i=0;i<MAX_LAYER_NB;i++) {
    if (!(l_layer[i].name).compare(name)) return i;
    if (!(l_layer[i].name).compare("")) {l_layer[i].name=name;return i;}
  }
  return -1;
}

// modify order to preserve dependencies ???
void reorder_layers(layer_t *l_layer, int layer_num) {
  string input = l_layer[layer_num].input;
  layer_t tmp;

  // extract the producer name from the input string
  input = input.substr(0, input.find("/"));
  
  for(int i = layer_num+1 ;i<MAX_LAYER_NB;i++) {
    if (!input.compare(l_layer[i].name)) {
      tmp = l_layer[i];
      l_layer[i] = l_layer[layer_num];
      l_layer[layer_num] = tmp;
    }
  }


}

int computeFilterSize(int in_size, int out_size, 
                      int padding, int stride, 
                      bool maxpool, int pooling_factor) {

// !!!!! This calculation is only valuable if the filter size has even numbers
int filter_size;
if (maxpool)
  filter_size= in_size + 2 * padding - (out_size * pooling_factor) * stride;
else
  filter_size = in_size + 2 * padding - (out_size) * stride + 1;

// For filter with odd numbers, the formula is the following: 
/*
if (maxpool)
  filter_size= in_size + 2 * padding - (out_size * pooling_factor  -1) * stride;
else
  filter_size = in_size + 2 * padding - (out_size -1) * stride;
*/
// Problem: how to know at this stage the filter is odd or even?? 
// We thus need to find another solution for getting the filter size.
return filter_size;
};



Status get_attributes_conv(const NodeDef& node, bool *relu,
                    bool *maxpool, int *pooling_factor, 
                    int *padding, int *strides,
                    int *maxpool_padding, int *maxpool_strides) {
  
  //Gets the attributes of a convolutional layer
  
  //Initialize requested parameter values
  Status status;
  Status status2(::tensorflow::error::UNIMPLEMENTED, "Padding is not supported by the GAP8 CNN library");
  *relu=false;
  *maxpool=false;
  *pooling_factor=0;
  *padding=0;
  *strides=0;
  *maxpool_padding=0;
  *maxpool_strides=0;
  string valid("VALID");

  // Define other values for some requested parameters, corresponding to the types under which 
  // they are stored in the protobuf file
  std::vector<int64> strides_value;
  std::vector<int64> pooling_value;
  std::vector<int64> maxpool_strides_value;
  string padding_string;
  string maxpool_padding_string;

  AttrSlice* attrslice = new AttrSlice(node);
  
  clog << "Getting Convolutional Layer attributes (get_attributes_conv)" << "\n";

  status=GetNodeAttr(*attrslice,"relu", relu);
  status=(GetNodeAttr(*attrslice,"maxpool",maxpool));
  status=(GetNodeAttr(*attrslice,"pooling_factor", &pooling_value));
  if (!status.ok())
     status=(GetNodeAttr(*attrslice,"pooling_factor",pooling_factor));
  else
    *pooling_factor=pooling_value[1];

  status=(GetNodeAttr(*attrslice,"padding", &padding_string));
  status=(GetNodeAttr(*attrslice,"strides", &strides_value));
  if (*maxpool) {
    status=(GetNodeAttr(*attrslice,"maxpool_padding", &maxpool_padding_string));
    status.Update(GetNodeAttr(*attrslice,"maxpool_strides", &maxpool_strides_value));
  }
  *strides=strides_value[1];

  if (padding_string.compare(valid) != 0)
    status.Update(status2);
  
  return status;
};

Status get_attribute(const NodeDef& node, bool *relu) {
  // Gets the "relu" attribute of a Dense layer
  Status status;
  *relu=false;
  AttrSlice* attrslice = new AttrSlice(node);
  
  status = GetNodeAttr(*attrslice,"relu", relu);
  return status;
};

void init_nphFile(ofstream& nphFile){

	nphFile << "#ifndef __CNNKERNEL_H__" << "\n";
	nphFile << "#define __CNNKERNEL_H__" << "\n";

	nphFile << "#include \"AutoTilerLibTypes.h\"" << "\n";
	nphFile << "#include \"CnnKernelsInit.h\"" << "\n";
	nphFile << "#include \"CNN_BasicKernels.h\"" << "\n";
	nphFile << "#define _L1_Memory_SIZE 38176" << "\n";
	nphFile << "#define _L2_Memory_SIZE 0" << "\n";
	nphFile << "extern char *L1_Memory; /* Size given for generation: 51200 bytes, used: 38176 bytes */" << "\n";
	nphFile << "extern char *L2_Memory; /* Size used for generation: 0 bytes */" << "\n" << "\n"<< "\n";
	}

void append_nphFile(ofstream& nphFile, int layerNb, bool convLayer=true) {

	if (convLayer) {
		nphFile << "extern void ConvLayer" << layerNb << "(" << "\n";
		nphFile << "short int * __restrict__ In," << "\n";
		nphFile << "short int * __restrict__ Filter," << "\n";
		nphFile << "short int * __restrict__ Bias," << "\n";
		nphFile << "short int * __restrict__ Out," << "\n";
		nphFile << "unsigned int Norm," << "\n";
		nphFile << "Kernel_Exec_T *Ker);" << "\n"<< "\n";

	} else {
		nphFile << "extern void Dense" << layerNb << "(" << "\n";
		nphFile << "short int * __restrict__ In," << "\n";
		nphFile << "short int * __restrict__ Filter," << "\n";
		nphFile << "short int * __restrict__ Bias," << "\n";
		nphFile << "short int * __restrict__ Out," << "\n";
		nphFile << "unsigned int Norm, " << "\n";
		nphFile << "unsigned int NormBias, " << "\n";
		nphFile << "Kernel_Exec_T *Ker); " << "\n"<< "\n";
	}
}

void initH2C2Files(ofstream& HFile, ofstream& CFile){
  // Writes first content of the main .h and .cc files
  // of the application GAP8 representation
  CFile << "#include \"weights_bias.h\"" << "\n";
  HFile << "#include \"rt/rt_api.h\"" << "\n";
  HFile << "#ifndef WEIGHTS_BIAS_H" << "\n";
  HFile << "#define WEIGHTS_BIAS_H" << "\n";

};

 
// Creates a vector filled up with all the layers of the Graph.
// The boolean ftp indicates that weights and bias parameters will be generated in floating point
void createLayersVector(std::vector<Layer*>& layers,
                        std::string& initWandB,
                        const std::string Gap8CodeDirName,
                        GraphDef graph,
                        std::string& l2_data,
                        layer_t *l_layer,
						bool ftp,
						ofstream& H2File,
						ofstream& C2File
                       ) {
  Status status;

  //initialize some parameters. Those numbers have been chosen to work for the 
  // cifar10 and mnist tutorials, but should not necessary work for other applications
  const int conv_norm_factor=14; // norm Factor to convert integer output to Q16 representation. Input was in Q16
  const int dense_norm_factor=16; // norm Factor for the Dense Layer output 
  const int dense_norm_bias = 13; // norm factor for the bias output. 
  // output  
  int lastOutputNumber = 0; //Nbre of output features of previous layer
  int lastOutputHeight = 0; //Height of each feature of previous layer(an image)
  int lastOutputWidth = 0;  // width of each feature of previous layer

  int OutputNumber = 0; //Nbre of output features of current  layer
  int OutputHeight = 0; //Height of each feature of current layer (an image)
  int OutputWidth = 0;  // width of each feature of current layer
  int i = 0;
  std::string myName = "";
  int fh=0;
  int fw=0;
  Color::Modifier red(Color::FG_RED);
  Color::Modifier def(Color::FG_DEFAULT);
  Color::Modifier orange(Color::FG_ORANGE);
  Color::Modifier green(Color::FG_GREEN);
  

  // Looping through the nodes of the application graph
  // !! We only treat linear graphs
    clog << "\n";
    clog << green << "****** Analysing Graph ******" << def << "\n";

    for (const NodeDef& node: graph.node() ){
      
      std::string name_layer = get_layer_name(node);
      clog << "\n";
      clog << "****** Analysing Layer "  << name_layer.c_str() << " ******" << "\n";

      int layer_num = get_layer_num(l_layer, name_layer);
      
      std::string op = get_node_operation(node);

      clog << "Node " << get_node_name(node).c_str() << "\n";
      clog << "Input " << get_node_input(node).c_str() << "\n";
      clog << "Operator " << op.c_str() << "\n";

      // We are testing every node operation
      if (!op.compare("MaxPool")){
        myName = "p" + std::to_string(i);
        settingOutputVariable(node, OutputNumber, OutputHeight,OutputWidth);
        layers.push_back(new PoolLayer(myName,i,  lastOutputNumber, 
                                      get_pooling_factor(node), OutputHeight, 
                                      OutputWidth, lastOutputHeight, 
                                      lastOutputWidth));
        settingLastOutputVariable(OutputNumber, OutputHeight, OutputWidth, 
                                  lastOutputNumber, lastOutputHeight, 
                                  lastOutputWidth);
        fill_layer_param_pool(l_layer, layer_num,get_pooling_factor(node));
        i += 1;
      }
      // implemented for treatment of a graph after GAP8 graph_transform functions
      // In this case, all conv2d operators have been fused with adjacent operators
      // into a GAP8_Conv2D operator with the following parameters
      
      // maxpool: with or without pooling
      // pooling_factor: Pooling factor if mawpool true
      // stride: jump of the window of the Conv2D treatment of the image
      else if(!op.compare("GAP8_Conv2D")){
        clog << "\n";
        clog << green << "Treating a node of type GAP8_Conv2D" << def << "\n";
        // Initialize some variables
        bool relu; // relu: with or without Relu
        bool maxpool; // relu: with or without Relu
        int pooling_factor;
        int padding;
        int stride;
        int maxpool_padding;
        int maxpool_strides;
        int filterWidth=0; // width of the GAP8_Conv2D filter
        int filterHeight=0; // Height of the GAP8_Conv2D filter
        char is_weights = 1; // set to true as the first Const input to the GAP8_Conv2D node contains the Weights parameters. 
                            // The second will contain the Bias parameters

        //get Convolutional layer attributes
        status=get_attributes_conv(node, &relu, &maxpool,&pooling_factor, &padding, &stride, &maxpool_padding, &maxpool_strides);
        if (!status.ok()) {
          tf2gap8Exception exc7(string("GAP8Conv2D operator attributes error\n") + status.error_message());
          throw exc7;
        }
        
        // Get node output parameters
        settingOutputVariable(node, OutputNumber, OutputHeight, OutputWidth);
        
        l_layer[layer_num].type=conv;

        // Loop on the inputs of the node
        for (const string& input : node.input()) {
            const NodeDef *test=FindNodebyName(input, graph);
            std::string op2 = get_node_operation(*test);
            clog << "input node " << input.c_str() << " Operator " << op2.c_str() <<  "\n";
                if (!op2.compare("Const")){
                    if (is_weights){
                      std::string name = Gap8CodeDirName + "/L2_W_" + std::to_string(i) + ".h";
                      fillHeaderFile(H2File,C2File, test ,i, is_weights, lastOutputNumber,
				     lastOutputWidth,lastOutputHeight,filterHeight, filterWidth,ftp);
		      clog << "filterHeight " << filterHeight << "\n";
		      clog << "filterWidth " << filterWidth << "\n";
                      is_weights = 0;
                    }
                    else{
                      std::string name = Gap8CodeDirName + "/L2_B_" + std::to_string(i) + ".h";
                      fillHeaderFile(H2File,C2File, test ,i, is_weights, lastOutputNumber,
				     lastOutputWidth,lastOutputHeight,fh,fw,ftp);
                    }  
                }
                else {
                  // this is the connection to the previous layer
                  l_layer[layer_num].input=get_node_name(*test);    
                }
          }// end of loop on the inputs
          myName = "c"+std::to_string(i);
          
          
          layers.push_back(new ConvLayer(myName, i,  lastOutputNumber, OutputNumber,
                                       filterWidth, filterHeight,1, relu,maxpool,pooling_factor,lastOutputHeight, 
                                       lastOutputWidth,conv_norm_factor));
          clog << "===> h " << OutputHeight << " w " << OutputWidth << " filterWidth " << filterWidth <<" filterHeight " << filterHeight << "\n";
          
          fill_layer_param_conv(l_layer, layer_num,OutputNumber, OutputHeight, OutputWidth,filterWidth,filterHeight,1);
          settingLastOutputVariable(OutputNumber, OutputHeight, OutputWidth, 
                                  lastOutputNumber, lastOutputHeight, 
                                  lastOutputWidth);
          i += 1;
          reorder_layers(l_layer, layer_num);
      }
      // Treating GAP8_DenseLayer with or without Relu (obtained after a Graph_transform fusion)
      else if (!op.compare("GAP8_DenseLayer")){
        int s = 0;
        bool relu;
        status= get_attribute(node,&relu); 
        clog << "\n";
        clog << green << "Treating a node of type GAP8_DenseLayer" << def << "\n";
        if (!status.ok()) {
          tf2gap8Exception exc8(string("GAP8DenseLayer operator attributes error\n") + status.error_message());
          throw exc8;
        }
	      int needless1, needless2;
        settingDenseOutputVariable(node, OutputNumber,  OutputHeight, OutputWidth);
        l_layer[layer_num].type=dense;
        for (const string& input : node.input()) {
                const NodeDef *test=FindNodebyName(input, graph);
		            std::string op2 = get_node_operation(*test);
                clog << " Input node " << input.c_str() << " Operator " << op2.c_str() << "\n";
                if (!op2.compare("Const")){
                    if (s == 1){
                      char Wt = 0;
                      std::string name = Gap8CodeDirName + "/L2_B_" + std::to_string(i) + ".h";
                      fillHeaderFile(H2File, C2File, test ,i, Wt, lastOutputNumber, lastOutputWidth,
				     lastOutputHeight,fh,fw,ftp);
                      s+=1;
                    }
                    else if (s == 2){
                      char Wt = 1;
                      std::string name = Gap8CodeDirName + "/L2_W_" + std::to_string(i) + ".h";
                      fillHeaderFile(H2File, C2File, test ,i, Wt, lastOutputNumber,
				     lastOutputWidth,lastOutputHeight,fh,fw,ftp);
                      clog << "Previous layer output width " << lastOutputWidth << "\n";
                      clog << "Previous layer output height " << lastOutputHeight << "\n"; 
                      clog << "Output Features nb " << OutputNumber << "\n";
                      clog << "Previous Layer Output Features nb " << lastOutputNumber << "\n";

                      fillDenseWeightTemplate(initWandB, l2_data, i, lastOutputWidth, 
                                              lastOutputHeight, OutputNumber, lastOutputNumber);
                      s+=1;
                    }
                    else{
                      s+=1;
                    }
                }

  	            else {
  		          // this is the connection to the layer connected to the input
  		          l_layer[layer_num].input=get_node_name(*test);		
  	            }
        }
        myName = "f"+std::to_string(i);
	// Already set previously ....
        //OutputNumber = get_output_number(node);
        layers.push_back(new DenseLayer(myName,i,  lastOutputNumber, OutputNumber,
                        -1, -1, lastOutputHeight, lastOutputWidth, 1, relu,dense_norm_factor,dense_norm_bias));
        fill_layer_param_dense(l_layer, layer_num,OutputNumber,1);
	// OutputHeight and OutputWidth have not been set for this dense layer ... why?
	settingLastOutputVariable(OutputNumber, OutputHeight, OutputWidth, 
                                  lastOutputNumber, lastOutputHeight, 
                                  lastOutputWidth);
        i += 1;
        reorder_layers(l_layer, layer_num);
      }
      else if (!op.compare("Placeholder")){
        clog << "\n";
        clog << green << "Treating a node of type PlaceHolder" << def << "\n";
	l_layer[layer_num].type=placeholder;
	settingOutputVariable(node, lastOutputNumber,lastOutputHeight,lastOutputWidth);
	l_layer[layer_num].out_width = lastOutputWidth;
        l_layer[layer_num].out_height = lastOutputHeight;
        l_layer[layer_num].n_out_feat = lastOutputNumber;
        reorder_layers(l_layer, layer_num);
      clog << green << "End of Treating a node of type PlaceHolder" << def << "\n";
      }
      else if (!op.compare("Reshape")){
        clog << "\n";
        clog << "Treating a node of type Reshape" << "\n";
	// We do not create a specific layer for it but just change the last output variables
	settingOutputVariable(node, lastOutputNumber,lastOutputHeight,lastOutputWidth);
      }
  }//end for    


  // Shouldn't we return the status? 

}// end create list of layers


// Calculates the biggest sizes for the network needed 
// for the calculation of the memory size we need on GAP8 IoT processor
void getMaxOneAndTwo(int& max1,
                     int& max2,
                     std::vector<Layer*>& layers
                   ){
     max1 = 0; 
     max2 = 0;
     bool b = true;
     tf2gap8Exception exc5(string("Error #tg5 in getMaxOneAndTwo: Pool Layer not handled for memory calculation if not fused with a Convolutional layer"));
     tf2gap8Exception exc6(string("Error #tg6 in getMaxOneAndTwo: Pool Layer not handled for memory calculation if not fused with a Dense layer"));
     // Looping through the layers vector
     for (std::vector<Layer*>::iterator it = layers.begin() ; it != layers.end(); ++it){
           std::string cls = (*it)->type;
           //dealing with a convolutional layer
           if(!cls.compare("conv")){
               ConvLayer* conv = (ConvLayer* ) *it;
               
               int filter_width = ((*conv).filter_width -1);
               int filter_height = ((*conv).filter_height -1);
               if((*conv).maxpool){
                  int tmp =  ((*conv).n_out_feat*((*it)->input_height - filter_height)*((*it)->input_width - filter_width))/ (2*(*conv).pooling_factor);
                  if(b){
                       max1 = (((tmp)>(max1))?(tmp):(max1));
                       b = false;
                  }
                  else{
                       max2 = (((tmp)>(max2))?(tmp):(max2));
                       b = true;
                  }
                }
                else{
                  int tmp =  ((*conv).n_out_feat*((*it)->input_height - 
                            filter_height)*((*it)->input_width - filter_width));
                  if(b){
                    max1 = (((tmp)>(max1))?(tmp):(max1));
                    b = false;
                  }  
                  else{
                    max2 = (((tmp)>(max2))?(tmp):(max2));
                    b = true;
                  }
                }
              }
              // Dense layer
              else if(!cls.compare("dense")){
                DenseLayer* dense = (DenseLayer* ) *it;
                //std::string next_cls = (*(it+1))->type;
                if( (it+1) != layers.end() && !(*(it+1))->type.compare("pool")){
                  throw exc6;

                  /*PoolLayer* pool = (PoolLayer* ) *(it+1);
                  int tmp =  ((*dense).n_out_feat*((*it)->input_height)*((*it)->input_width))/ 
                            (2*((*pool).pooling_factor));
                  if(b){
                    max1 = (((tmp)>(max1))?(tmp):(max1));
                    b = false;
                  }
                  else{
                    max2 = (((tmp)>(max2))?(tmp):(max2));
                    b = true;
                  }*/
                }
                else{
                  int tmp =  ((*dense).n_out_feat*((*it)->input_height)*((*it)->input_width));
                  if(b){
                    max1 = (((tmp)>(max1))?(tmp):(max1));
                    b = false;
                  }
                  else{
                    max2 = (((tmp)>(max2))?(tmp):(max2));
                    b = true;
                  }
                }
              }

     }

}


void createDefine(std::vector<Layer*>& layers,
                  std::string& tmp,
                  int max1,
                  int max2){

    for (std::vector<Layer*>::iterator it = layers.begin() ; it != layers.end(); ++it){
        std::string cls = (*it)->type;
        if(!cls.compare("conv")){
            ConvLayer* test = (ConvLayer*) *it;
            if ((it+1)==layers.end())
              tmp = tmp + "#define CLast"  +
                 "_NFEAT " + std::to_string((*test).n_out_feat) + "\n";
            else
              tmp = tmp + "#define C" + std::to_string((*it)->layer_number) +
                 "_NFEAT " + std::to_string((*test).n_out_feat) + "\n";

        }
        else if(!cls.compare("dense")){
            DenseLayer* test = (DenseLayer* ) *it;
            if ((it+1)==layers.end())
              tmp = tmp + "#define CLast" + 
                  "_NFEAT " + std::to_string((*test).n_out_feat)+ "\n";
            else
              tmp = tmp + "#define C" + std::to_string((*it)->layer_number) + 
                  "_NFEAT " + std::to_string((*test).n_out_feat)+ "\n";
        }
    }
    std::vector<Layer*>::iterator it = layers.begin() ;
    tmp = tmp + "#define IMG_SIZE " + std::to_string((*it)->input_height) +"\n";
    tmp = tmp +  "#define BUF0_SIZE " + std::to_string(max1) + "\n";
    tmp = tmp + "#define BUF1_SIZE " + std::to_string(max2) + "\n";
    tmp = tmp +  "\n";
}

std::string headerConvLayer(ConvLayer* test,
                            std::string name,
                            std::string kernel,
                            int &tot_size_coeff,
                            int &tot_size_bias){
  std::string tmp = "{";
  //tmp = tmp + "#define NAME \"" + name + "\"\n";
  //tmp = tmp + "#define KERNEL\"" + kernel + "\"\n";
  tmp = tmp + std::to_string(test->n_in_feat) + ",";
  tmp = tmp + std::to_string(test->n_out_feat) + ",";
  tmp = tmp + std::to_string(test->input_height) + ",";
  tmp = tmp + std::to_string(test->input_width) + ",";
  tmp = tmp + std::to_string(test->filter_width) + "," ;
  tmp = tmp + std::to_string(test->filter_height) + "," ;
  tmp = tmp + std::to_string(test->relu) + ",";
  tmp = tmp + std::to_string(test->maxpool) + ",";
  tmp = tmp + std::to_string(test->pooling_factor) + ",";
  tmp = tmp + std::to_string(test->norm_factor);
  tmp = tmp + "}," + "\n";
  tot_size_coeff += test->n_out_feat * test->n_in_feat * test->filter_height * test->filter_width;
  tot_size_bias += test->n_out_feat;
  clog << "Conv layer coeff" +  std::to_string(tot_size_coeff) + " bias " + 
          std::to_string(tot_size_bias) + "\n";
  return tmp;

}

std::string headerDenseLayer(DenseLayer* test,
                            std::string name,
                            std::string kernel,
                            int &tot_size_coeff,
                            int &tot_size_bias){
  
  Color::Modifier green(Color::FG_GREEN);
  Color::Modifier def(Color::FG_DEFAULT);
  
  std::string tmp = "{";
  tmp = tmp + std::to_string(test->n_in_feat) + ",";
  tmp = tmp + std::to_string(test->n_out_feat) + ",";
  tmp = tmp + std::to_string(test->input_height) + ",";
  tmp = tmp + std::to_string(test->input_width) + ",";
  tmp = tmp + std::to_string(test->relu) + ",";
  tmp = tmp + std::to_string(test->norm_factor) + ",";
  tmp = tmp + std::to_string(test->norm_bias);
  tmp = tmp + "}," + "\n";
  tot_size_coeff += test->n_out_feat * test->n_in_feat * test->input_width * test->input_height;
  tot_size_bias += test->n_out_feat;
 
  clog << green << "Dense Layer Info"  <<  def << "\n";
  clog << "Total size of weights: " << std::to_string(tot_size_coeff) << "\n";
  clog << "Total size of bias: "  << std::to_string(tot_size_bias)  << "\n";
  clog << "Number of output features: " << std::to_string(test->n_in_feat)  << "\n";
  clog << "Number of input features: " << std::to_string(test->n_in_feat)  << "\n";
  clog << "Input Width: " << std::to_string( test->input_width) << "\n";
  clog << "Input Height: " << std::to_string( test->input_height) << "\n";

  return tmp;

}

// Processes the list of Layers previously created in process_graph and generates
// the corresponding GAP8 code
void createForLoopandHeader(std::vector<Layer*>& layers,
                            std::string& str,
                            string Gap8CodeDirName,
                            int &tot_size_coeff,
                            int &tot_size_bias,
                            ofstream& nphFile,
                            ofstream& H2File,
                            ofstream& C2File){
  str = "\n";
  int function_num = 0;
  bool first = true;
  bool alternate = false;
  std::string function_name;
  std::string tmp, tmp_conv, tmp_dense;
  std::string kernel;
  ofstream Hfile;

  Color::Modifier orange(Color::FG_ORANGE);
  Color::Modifier green(Color::FG_GREEN);
  Color::Modifier def(Color::FG_DEFAULT);
  
  Hfile.open(Gap8CodeDirName + "/param_layers.h");
  tmp_conv= tmp_conv + "struct param_conv_layer convLayers[] = {";
  tmp_dense= tmp_dense + "struct param_dense_layer denseLayers[] = {";
  int idx=0;
  int nb_conv=0; // nb of convolutional layers
  int nb_dense=0; // nb of dense layers

  clog << "\n";

  // initialize network_process.h file
  init_nphFile(nphFile);

  clog << green << "****** Generating GAP8 Code ******" << def << "\n";
  for (std::vector<Layer*>::iterator it = layers.begin() ; it != layers.end(); ++it){
      std::string cls = (*it)->type;
      clog << "Generating code for a layer of type: " << ((*it)->type).c_str() << "\n";

      if(!cls.compare("conv")){
          nb_conv++;
          ConvLayer* test = (ConvLayer *) *it;
      
          tmp = "ConvLayer" + std::to_string(nb_conv);
          kernel = tmp;
          tmp_conv=tmp_conv + headerConvLayer(test, tmp, kernel,tot_size_coeff,tot_size_bias);
          //Hfile.close();
          tmp = "     " + tmp;
          str = str + tmp  +"(";
          // if first call, l2_x et sortie dans l_big0 ou l_big1 alternativement
          std::string inBuf, outBuf;
          if (first){
            inBuf="l2_x";
            outBuf="l2_big0";
            first = false;
            alternate = true;
          }
          else{
            if (alternate){
              inBuf="l2_big0";
              outBuf ="l2_big1";
              alternate = false;
            }
            else{
              inBuf="l2_big1";
              outBuf ="l2_big0";
              alternate = true;
            }
          }
            // First Append network_process.h file with convLayer definition function
          	// There is one new function def per conv Layer as it is going to be inlined
          	append_nphFile(nphFile, nb_conv,true);
            str = str + inBuf + ",L2_W_"+ std::to_string((*it)->layer_number);
            str=str + ",L2_B_" + std::to_string((*it)->layer_number); // Bias
            str = str + "," + outBuf + ",";  // out 
            str = str + std::to_string(QF);  // Norm
            str=str + ",AllKernels + " + std::to_string(idx) + "); \n"; // kernel
            
          function_num += 1;
          
      }

      else if(!cls.compare("dense")){
          nb_dense++;
          DenseLayer* test = (DenseLayer*) *it;
          tmp= "Dense" + std::to_string(nb_dense);
          tmp_dense=tmp_dense + headerDenseLayer(test, tmp, kernel,tot_size_coeff,tot_size_bias);
          tmp = tmp +  "(";
          // First Append network_process.h file with denseLayer definition function
          // There is one new function def per conv Layer as it is going to be inlined
          append_nphFile(nphFile, nb_dense,false);
          if(alternate){
	        tmp = tmp + "l2_big0,L2_W_"+std::to_string((*it)->layer_number); // in + filter
	          
	        tmp=tmp+",L2_B_" + std::to_string((*it)->layer_number); // Bias
	        tmp=tmp+ ",l2_big1"; // out
	        //tmp=tmp + std::to_string((test)->n_out_feat);
	        tmp=tmp + ",16"; // Norm 
	        tmp=tmp+ ",13"; // Norm Bias
	         
	        tmp=tmp +",AllKernels + " + std::to_string(idx) + "); \n"; // Kernel
	        alternate = false;
          }
          else{
            tmp= tmp + "l2_big1,L2_W_"+std::to_string((*it)->layer_number); // in + filter
            tmp= tmp + ",L2_B_" + std::to_string((*it)->layer_number); // Bias
            tmp= tmp + ",(int*)l2_big0"; // out
            tmp= tmp + ",16"; // Norm 
            tmp= tmp + ",13"; // Norm Bias
            //tmp=tmp+ std::to_string((test)->n_out_feat);
            tmp=tmp + ",AllKernels + " + std::to_string(idx) + "); \n";
            alternate = true;
          }
          function_num += 1;
          str = str +  "   " + tmp;
      }
      else
	cerr << orange << "Warning: Only conv and dense layers, eventually fused with Maxpool and Relu are supported by this bridge up to now." << "\n"; 
      idx++;
  }
  Hfile << "#define NB_CONV " << nb_conv << "\n";
  Hfile << "#define NB_DENSE " << nb_dense << "\n";
  Hfile << tmp_conv << "};\n";
  Hfile << tmp_dense << "};\n";  
  str = str + "} \n\n";
  Hfile.close();
}

// Processes the list of Layers previously created in process_graph and generates
// the corresponding GAP8 code
// This function is not used. Keeping it for future investigation
/*void createForLoopandHeader_new(std::string& str,
                            string Gap8CodeDirName,
                            int &data_alive){
  str = "\n";
  int function_num = 0;
  bool first = true;
  //bool alternate = false;
  std::string function_name;
  std::string tmp;
  std::string kernel;
  int sizeout;
  
  int input_layer_nb;
  layer_t* it;
  for (int i=0;i<MAX_LAYER_NB;i++){
    it=l_layer+i;
    if (!(*it).name.compare("")) break;
    sizeout = (*it).n_out_feat *  (*it).out_width  *  (*it).out_height;
    // allocate (except for inputs)
    if ((input_layer_nb=get_input_layer_nb(l_layer, i))!=-1)
      str = str + "    buf[" + std::to_string(i) + "]=malloc(" +  
    std::to_string(sizeout) + "*sizeof(Word16));\n" ;
    data_alive += sizeout;
    if (data_alive>maxsizedata) maxsizedata=data_alive;

    type_layer cls = (*it).type;
      if(cls  == conv){
          ofstream Hfile;
          layer_t test =  *it; // layer
          str = str + "   for(of=0; of<" + std::to_string(test.n_out_feat)+
               "; of++){ \n" + "  //printf(\"compute output feature %d\\n\",of) \n";
          int out_height = (test.out_height);
          int out_width = (test.out_width);
          tmp = "Conv"+std::to_string(test.filter_width)+"x"+std::to_string(test.filter_height);
          if(test.activation_relu){
              tmp = tmp + "ReLU";
          }
          if( (*it).activation_pool){
              layer_t pool =  *it;
              tmp = tmp + "MaxPool" + std::to_string(pool.pooling_factor) + "x" + 
              std::to_string(pool.pooling_factor);
              //out_width = out_width/pool.pooling_factor;
              //out_height = out_height/pool.pooling_factor;
          }
          kernel = tmp;
          tmp = tmp +"_"+std::to_string(function_num);
          function_name = Gap8CodeDirName + "/" + tmp + ".h";
          Hfile.open(function_name.c_str());
          Hfile << headerConvLayer_(test, tmp, kernel,tot_size_coeff,tot_size_bias);
          Hfile.close();
          tmp = "     " + tmp;
          str = str + tmp  +"(";
          if (first){
            str = str + "l2_x,L2_W_"+std::to_string(i)+"+of*"+
            std::to_string((test).filter_width)+"*"+
            std::to_string((test).filter_height)+"*"+std::to_string((test).n_in_feat);
            str = str + ",buff[" + std::to_string(i) + "]" + " +of*"+
            std::to_string(out_height)+"*"+std::to_string(out_width) + 
            ","+std::to_string(QF)+",L2_B_";
            str = str +std::to_string(i)+"[of],NULL); \n";
            first = false;
            //alternate = true;
          }
          else{
	           str = str + "buff[" + std::to_string(input_layer_nb) + "],L2_W_"+std::to_string(i)+"+of*"+
              std::to_string((test).filter_width)+"*"+std::to_string((test).filter_height)+
              "*"+std::to_string((test).n_in_feat);
	           str = str + ",buff[" + std::to_string(i) + "]+of*"+std::to_string(out_height)+
              "*"+std::to_string(out_width) +  ","+std::to_string(QF)+",L2_B_";
	           str = str +std::to_string(i)+"[of],NULL); \n";
	           //alternate = false;
	    
          }
          function_num += 1;
          str = "   " + str + "\n }\n";
      }

      else if(cls == dense){
          ofstream Hfile;
	        layer_t test =  *it;
          tmp = "LinearLayer";
          if(test.activation_relu)
              tmp = tmp + "ReLU";
          kernel = tmp;
          tmp = tmp + "_" +std::to_string(function_num);
          function_name = Gap8CodeDirName + "/" + tmp+".h";
          Hfile.open(function_name.c_str());
          Hfile << headerDenseLayer_(test, tmp, kernel,tot_size_coeff,tot_size_bias);
          Hfile.close();
          tmp = tmp +  "(";
	        tmp = tmp + "buff[" + std::to_string(input_layer_nb) + "],0,L2_W_"+std::to_string(i)+",16,L2_B_"+
          std::to_string(i)+",10" +",buff[" + std::to_string(i) + "]"+
          std::to_string((test).n_out_feat)+",NULL); \n";
	        //alternate = false;
          function_num += 1;
          str = str +  "   " + tmp;
      }
      input_layer_nb=get_input_layer_nb(l_layer, i)  ;
      // dont free the input! (input_layer_nb==0) 
      if (input_layer_nb!=-1 && input_layer°nb && !still_alive(l_layer, input_layer_nb,i)) {
	     str = str + "    " + "free(buf[" + std::to_string(input_layer_nb) + "]);\n" ;
	     data_alive = data_alive - l_layer[input_layer_nb].n_out_feat*l_layer[input_layer_nb].out_width*
                l_layer[input_layer_nb].out_height;
      }
  }

  str = str + "} \n\n";
  }*/


// Adds a fixed code file content to the GAP8 code being generated
void add_file(std::ofstream& oFile, std::string fixedFileName, std::string TF2GAP8_Dir) {
  Color::Modifier red(Color::FG_RED);
  Color::Modifier def(Color::FG_DEFAULT);
  
  ifstream file( TF2GAP8_Dir + fixedFileName, ios::in);
    if (file){
      std::string ligne;
      while(getline(file, ligne)){
          oFile << ligne << endl;
      }
    }
    else 
      cerr << red << "Error: Missing file " << TF2GAP8_Dir.c_str() << "/" << fixedFileName.c_str() << def << "\n";
};


// dumps all the layers array for debuging purposes
void dump_layers(layer_t *l_layer) {
        clog << "\n"; 
	fprintf(stderr,"Dump layers\n");
	    for(int i=0;i<MAX_LAYER_NB;i++) 
	      if ((l_layer[i].name).compare("")) {
	    	   fprintf(stderr,"Layer %s\n",l_layer[i].name.c_str());
	    	   fprintf(stderr,"\ttype %s\n",(l_layer[i].type==conv)?"conv":
	            (l_layer[i].type==dense)?"dense":"placeholder");
	    	   fprintf(stderr,"\tinput %s\n",l_layer[i].input.c_str());
	    	   fprintf(stderr,"\tnof %d ow %d oh %d\n",l_layer[i].n_out_feat,l_layer[i].out_width,
	            l_layer[i].out_height);
	    	   fprintf(stderr,"\tnif %d iw %d ih %d\n",l_layer[i].n_in_feat,l_layer[i].input_width,
	            l_layer[i].input_height);
		
	      }else break;
	    
	    fprintf(stderr,"\n");
};

int main(int argc, char* argv[]) {
    // This program has to be launched with four parameters:
    string graphFile; // full path of the file storing the training gaph
    string graphDirName; // full path of the directory were the graphFiles is stored
    string TF2GAP8_Dir; // Source directory of TF2GAP8
    string floating_point; // true: the bias and weights are generated in floating point else, they are quantized
    
    int tot_size_data = 0;
    int tot_size_coeff = 0;
    int tot_size_bias = 0;
    layer_t l_layer[MAX_LAYER_NB];
    string false_str("false");
    string ftp_str;
    bool ftp=false;
    
    // Define some color modes
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);

    //test arguments
    if (argc>0)
        graphFile=std::string(argv[1]);
       else {
        std::cerr << red << "Error: Missing Graph protobuf file as first argument" << def  << "\n";
        return 1;
       }
    if (argc>1)
        graphDirName=std::string(argv[2]);
      else
        graphDirName="/tmp";
    if (argc>2)
      TF2GAP8_Dir=std::string(argv[3]);
    else
      TF2GAP8_Dir=std::string("~/tensorflow/TF2GAP8");
    
    if (argc>3){
      clog << "argv4 " << argv[4] << "\n";
      ftp_str=std::string(argv[4]);
      if (ftp_str.compare(false_str)==0)
	ftp=false;
      else
	ftp=true;
    }
    else
      ftp=false;
    
    // Print program input parameters
    clog << green << "****** Input Parameters ******" << def << "\n";
    clog << "graph File " << graphFile << "\n";
    clog << "graph Directory: " << graphDirName << "\n";
    clog << "tf2gap8 source code Directory: " << TF2GAP8_Dir << "\n";
    clog << "Floating_point generation: " << ftp << "\n";
    clog << green << "****** END Input Parameters ******" << def << "\n";

    // deduct some other argumets
    const string graphFilePath=graphDirName + '/'+ graphFile;
    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
      std::cerr << red << "Error: " << status.ToString() << def << "\n";
      return 1;
    }

    GraphDef graph; // variable to hold the  training graph

    //checking if graphDir exists, exit otherwise
    status=dirExists(graphDirName);
    if (!status.ok()) {
      std::cerr << red << "Error: " << status.ToString() << def << "\n";
      return 1;
    }

    //create all necessary directories to store GAP8 generated code
    const string GAP8CodeDirName=graphDirName + "/" + "GAP8Code";

    std::string l2_data = "";
    std::string initWandB = "";

    // create GAP8 generated code directory
    createGAP8CodeDir(GAP8CodeDirName);

    // Read Graph from binary protobuf file
    status= ReadBinaryProto(Env::Default(),graphFilePath, &graph);
    if (!status.ok()) {
      std::cerr << red << status.ToString() << def << "\n";
      return 1;
    }
    // Prints the graph for debug purposes. Can be commented 
    // if not needed
    /*ofstream graphDumpFile;
    graphDumpFile.open("graphDump.txt");
    graphDumpFile << graph.DebugString() << "\n";
    graphDumpFile.close();
*/
    // open GAP8 main code File
    ofstream CFile; // GAP8 network process main file
    ofstream DFile;  //GAP8 defines file
    ofstream nphFile; 
    CFile.open(GAP8CodeDirName + "/network_process.c");
    DFile.open(GAP8CodeDirName + "/define.h");
    nphFile.open(GAP8CodeDirName + "/network_process_proto.h");

    // define .h and .c files for the generation weights & bias of GAP8 code
  	ofstream H2File;
  	ofstream C2File;
  	H2File.open(GAP8CodeDirName + "/weights_bias.h");
  	C2File.open(GAP8CodeDirName + "/weights_bias.c");
  	// Initialize H2File and C2File
  	initH2C2Files(H2File,C2File);

    //Create an arbitrary vector  of layers to be filled out when reading the graph
    std::vector<Layer*> layers;
    std::string tmp = "";
    int max1;
    int max2;
	try {
    	// Fill up layers vector with graph information
        // This can throw an exception if a problem occurs
	  createLayersVector(layers,initWandB,GAP8CodeDirName,graph,l2_data,l_layer,ftp,H2File,C2File);  

    	// dump vector of layers for debug purposes. 
    	// can be commented out if not needed. 
      /*clog << green << "****** Dump Effective Layers ******" << def << "\n";
    	dump_layers(l_layer);
      clog << green << "****** END Dump Effective Layers ******" << def << "\n"; 
      */
    	// Process the layers vector and generate the 
    	// GAP8 code corresponding to the protobuf GRAPH
    
      // Calculates the biggest size for the network
      getMaxOneAndTwo(max1, max2, layers);   

      clog << "\n" << "max1 "  << max1 << " max2 " << max2 << "\n";                           
      // Write the "define" part  of the C File
      createDefine(layers, tmp, max1, max2);                             
      tot_size_data = max1 + max2;
      DFile << tmp;

      CFile << "#include \"network_process.h\"\n\n";
      CFile << "void network_process () { ";
      createForLoopandHeader(layers, tmp, GAP8CodeDirName, tot_size_coeff,tot_size_bias,nphFile,H2File, C2File);
      CFile << tmp << "\n";
  
      std::vector<Layer*>::iterator it = layers.end()-1;
    }   
    catch (tf2gap8Exception e) {
     	  cerr << e.what() << endl;
   	  	// catch block
      	CFile.close(); // Close GAP8 network process code file
      	DFile.close(); // Close GAP8 Define code file
      	nphFile << "#endif" << "\n";
      	nphFile.close(); // close network_process.h file
      	return EXIT_FAILURE; 
    } 

    CFile.close(); // Close GAP8 network process code file
    DFile.close(); // Close GAP8 Define code file
    nphFile << "#endif" << "\n";
    nphFile.close(); // close network_process.h file
    H2File << "#endif\n";
  	H2File.close();
  	C2File.close();

    clog << "\n";
    clog << green << "****** GAP8 Code Generation Completed Successfully ******" << def << "\n";
    clog << "\n";
    clog << green << "****** Estimated Memory Footprint (assuming all data in 16 bits)" << def << "\n"; 
    clog << "data: " << tot_size_data * 2 << " Bytes" << "\n"; 
    clog << "Weights: " << tot_size_coeff * 2 <<  " Bytes" << "\n"; 
    clog << "Bias: " << tot_size_bias * 2 <<  " Bytes" << "\n"; 
    clog << "Total: " << (tot_size_coeff + tot_size_bias + tot_size_data)*2 <<  " Bytes" << "\n"; 
    
    return 0;
}
