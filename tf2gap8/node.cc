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
This file contains all the usefull --or might be usefull-- functions for nodes
Contributor: Nicolas Lepetit, Corine Lamagdeleine, Joel Cambonie
*/
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "node.h"
#include "colormod.h"
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

using namespace std;


std::string ReplaceAll(std::string str, const std::string& from, const std::string& to) {
  size_t start_pos = 0;
  while((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
  }
  return str;
}


std::string get_node_name(const tensorflow::NodeDef &node){
    return node.name();
}

std::string get_node_operation(const tensorflow::NodeDef &node){
    return node.op();
}

std::string get_node_type(const tensorflow::NodeDef &node){
    auto finder = node.attr().find("T");
    return SummarizeAttrValue(finder->second);
    
}

std::string get_node_dtype(const tensorflow::NodeDef &node){
    auto finder = node.attr().find("dtype");
    return SummarizeAttrValue(finder->second);
}

std::string get_node_output_shapes(const tensorflow::NodeDef &node){
    cerr << "get_node_output_shapes" << "\n";
    auto finder = node.attr().find("_output_shapes");
    std::string shape = SummarizeAttrValue(finder->second);
    cerr << "return shape **" << shape << "**" <<  "\n";
    return shape;
}

std::string get_conv_padding(const tensorflow::NodeDef &node){
    Color::Modifier red(Color::FG_RED);
    Color::Modifier def(Color::FG_DEFAULT);
    std::string verify = get_node_operation(node);
    if( verify.compare("Conv2D")==0){
        auto finder = node.attr().find("padding");
        return SummarizeAttrValue(finder->second);
    }
    else{
      std::cerr << red << "Error: in node.cc in get_conv_padding: " << get_node_name(node) << " is not a Conv Layer" << def <<"\n";
        return "";
    }
}

std::string get_conv_strides(const tensorflow::NodeDef &node){
    Color::Modifier red(Color::FG_RED);
    Color::Modifier def(Color::FG_DEFAULT);
    std::string verify = get_node_operation(node);
    if( verify.compare("Conv2D")==0){
        auto finder = node.attr().find("strides");
        return SummarizeAttrValue(finder->second);
    }
    else{
        std::cerr << red << "Error: in node.cc in get_conv_strides:" << get_node_name(node) << " is not a Conv Layer" << def <<"\n";
        return "";
    }
}

std::string get_maxpool_strides(const tensorflow::NodeDef &node){
    Color::Modifier red(Color::FG_RED);
    Color::Modifier def(Color::FG_DEFAULT);
    std::string verify = get_node_operation(node);
    if( verify.compare("MaxPool")==0){
        auto finder = node.attr().find("strides");
        return SummarizeAttrValue(finder->second);
    }
    else{
        std::cerr << red << "Error: in node.cc in get_maxpool_strides " << get_node_name(node) << " is not a MaxPool Layer" << def <<"\n";
        return "";
    }
}

std::string get_maxpool_padding(const tensorflow::NodeDef &node){
    Color::Modifier red(Color::FG_RED);
    Color::Modifier def(Color::FG_DEFAULT);
    std::string verify = get_node_operation(node);
    if( verify.compare("MaxPool")==0){
        auto finder = node.attr().find("padding");
        return SummarizeAttrValue(finder->second);
    }
    else{
        std::cerr << red << "Error: in node.cc in get_maxpool_padding " << get_node_name(node) << " is not a MaxPool Layer" << def <<"\n";
        return "";
    }
}

std::string get_maxpool_ksize(const tensorflow::NodeDef &node){
    Color::Modifier red(Color::FG_RED);
    Color::Modifier def(Color::FG_DEFAULT);
    std::string verify = get_node_operation(node);
    if( verify.compare("MaxPool")==0){
        auto finder = node.attr().find("ksize");
        return SummarizeAttrValue(finder->second);
    }
    else{
      std::cerr << red << "Error: in node.cc in get_maxpool_ksize " << get_node_name(node) << " is not a MaxPool Layer" << def <<"\n";
      return "";
    }
}

std::string get_GAP8_conv2d_pooling(const tensorflow::NodeDef &node){
    Color::Modifier red(Color::FG_RED);
    Color::Modifier def(Color::FG_DEFAULT);
    cerr << "get_GAP8_conv2d_pooling" << "\n";
    std::string verify = get_node_operation(node);
    cerr << "node operation: " << verify << "\n";
    if( verify.compare("GAP8_Conv2D")==0){
        auto finder = node.attr().find("pooling_factor");
        return SummarizeAttrValue(finder->second);
    }
    else{
      std::cerr << red << "Error: in node.cc in get_GAP8_conv2d_pooling " << get_node_name(node) << " is not a Conv Layer" << def <<"\n";
        return "";
    }
}

std::string get_node_input(const tensorflow::NodeDef &node){
    std::string ret = "(";
    bool first = true;
    for (const string& input : node.input()) {
        if (!first){
            ret.push_back(','); //", ");
        }
        first = false;
        ret.append(input); //(&ret, input);
    }
    ret.push_back(')');
    return ret;
}


int get_output_number(const tensorflow::NodeDef &node){
   // Getting the number of output features of the node
    std::string temp = get_node_output_shapes(node);
    if (temp.length()>4) {
      int i = temp.length()-3;
      std::string out = "";
      out.push_back(temp[i]);
      while(temp[i-1]!=','){
          out.insert(0,1,temp[i-1]);
          i = i - 1;
      }
      return std::stoi(out);
    }
    else
      return 0;
}

int get_output_width(const tensorflow::NodeDef &node){
  std::string temp = get_node_output_shapes(node);
  if (temp.length()>4) {
    int i = temp.length()-3;
    while(temp[i-1]!=','){
        i = i - 1;
      }
    i = i - 1;
    std::string out = "";
    out.push_back(temp[i]);
    while(temp[i-1]!=','){
        out.insert(0,1,temp[i-1]);
        i = i - 1;
    }
    return std::stoi(out);
  }
  else
    return 0;
}

int get_output_height(const tensorflow::NodeDef &node){
  std::string temp = get_node_output_shapes(node);
   if (temp.length()>4) {
    int i = temp.length()-3;
    while(temp[i-1]!=','){
        i = i - 1;
      }
    i = i - 1;
    while(temp[i-1]!=','){
        i = i - 1;
      }
    i = i - 1;
    std::string out = "";
    out.push_back(temp[i]);
    while(temp[i-1]!=','){
        out.insert(0,1,temp[i-1]);
        i = i - 1;
    }
    return std::stoi(out);
  } 
  else
    return 0; 
}


void settingOutputVariable(const tensorflow::NodeDef &node,
		    int& n_out_feat,
		    int& output_height,
		    int& output_width){
  Color::Modifier orange(Color::FG_ORANGE);
  Color::Modifier def(Color::FG_DEFAULT);
  std::string shapeStr = get_node_output_shapes(node);

  n_out_feat=0;
  output_width=0;
  output_height=0;

  string s1 = ReplaceAll(shapeStr, string("["), string(""));
  string s2 = ReplaceAll(s1,string("]"), string(""));
  string str = ReplaceAll(s1,string("?,"), string("")); 
  clog << s1 << "\n";
  clog << s2 << "\n";
  clog << str << "\n";

  std::vector<int> vect;

  std::stringstream ss(str);

  int i;

  while (ss >> i)
    {
      vect.push_back(i);

      if ((ss.peek() == ',') || (ss.peek() == '?'))
	ss.ignore();
    }

  for (i=0; i< vect.size(); i++)
    std::clog << vect.at(i)<<std::endl;

  if (vect.size()>0)  {
    n_out_feat=vect.at(abs(vect.size()-1));
  }
  else
    cerr << orange << "Warning: node " << get_node_name(node).c_str() << " with operator " << get_node_operation(node).c_str() << " has an empty shape: " << shapeStr.c_str() << def << "\n"; 
  if (vect.size()>1)  {
      output_width=vect.at(abs(vect.size()-2));
  } else
      cerr << orange << "Warning: node " << get_node_name(node).c_str() << " with operator " << get_node_operation(node).c_str() << " has no output width value in shape: " << shapeStr.c_str() << def << "\n";
  if (vect.size()>2)  {
      output_height=vect.at(abs(vect.size()-3));
  } else
      cerr << orange << "Warning: node " << get_node_name(node).c_str()<< " with operator " << get_node_operation(node).c_str() << " has no output height value in shape: "<<shapeStr.c_str() << def << "\n";

}


int get_pooling_factor(const tensorflow::NodeDef &node){
  std::string temp = get_maxpool_ksize(node);
  if (temp.length()>4) {
    int i = temp.length()-3;
    while(temp[i-1]!=','){
        i = i - 1;
      }
    i = i - 1;
    std::string out = "";
    out.push_back(temp[i]);
    while(temp[i-1]!=','){
        out.insert(0,1,temp[i-1]);
        i = i - 1;
    }
    return std::stoi(out);
  }
  else
    return 0;
}

int get_pooling_factor_conv2d(const tensorflow::NodeDef &node){
  //std::string temp = get_maxpool_ksize(node)
  cerr << "get_pooling_factor_conv2d" << "\n";
  std::string temp = get_GAP8_conv2d_pooling(node);
  cerr << "temp= " << temp << "\n";
  int i = temp.length()-3;
  if (temp.length()>4) {
    while(temp[i-1]!=','){
        i = i - 1;
      }
    i = i - 1;
    std::string out = "";
    out.push_back(temp[i]);
    while(temp[i-1]!=','){
        out.insert(0,1,temp[i-1]);
        i = i - 1;
    }
    return std::stoi(out);
  }
  else
    return 0;
}
