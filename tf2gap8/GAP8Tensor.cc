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
contributors: Nicolas Lepetit, Corine Lamagdeleine
*/
#include "GAP8Tensor.h"
#include <math.h>       /* pow */
using namespace tensorflow;

int N = 14; // BITS NUMBERS
float Q = pow(2, N)-1;

// We need to explain that shapes in Tensorflow and GAP8 Library are
// not defined in the same order. So we need to change the way
// tensors values are stored to fit GAP8 expectations
template <typename T>
string GAP8SummarizeArray(int64 limit,
			  int64 num_elts,
			  const TensorShape& tensor_shape,
			  const char* data,
			  int last_out,
			  int last_w,
			  int last_h,
			  bool ftp) {
  string ret;
  const T* array = reinterpret_cast<const T*>(data);

  const gtl::InlinedVector<int64, 4> shape = tensor_shape.dim_sizes();

  if (shape.empty()) {
    for (int64 i = 0; i < limit; ++i) {
      if (i > 0) strings::StrAppend(&ret, ",");
      strings::StrAppend(&ret, array[i]);
    }
    if (num_elts > limit) strings::StrAppend(&ret, "...");
    return ret;
  }

  int64 data_index = 0;
  const int shape_size = tensor_shape.dims();
  int64 size = 1;
  for (int i = 0; i < shape_size; i++){
      size = size*shape[i];
  }
  float max = array[0];
  float min = array[0];
  for (int j = 0; j < size; j++){
    if (array[j]<min){
      min = array[j];
    }
    if (array[j]>max){
      max = array[j];
    }
  }

  if (shape_size == 4){
    // It means that we've a 4D tensor with this arrengment [Hfil][Wfil][n_in][n_out]
    int dimHfilter = shape[0];
    int dimWfilter = shape[1];
    int dimNin = shape[2];
    int dimNout = shape[3];
    int filter = dimWfilter*dimHfilter;
    int offset = dimNout*dimNin;
    strings::StrAppend(&ret,"    ");
    for(int l = 0; l < dimNout; l++){
        for(int t = 0; t < dimNin; t++){
            for(int k = l + t*dimNout; k < size; k+=offset){
              // Weights and Bias are either generated in floating point or integer
              if (ftp){
		float tmpf = array[k];
	        strings::StrAppend(&ret, tmpf,", ");
	      }
	      else {
		int tmpq  = Q*(array[k]);
		strings::StrAppend(&ret, tmpq,", ");
	      }
            }
            strings::StrAppend(&ret, "\n    ");
        }
        strings::StrAppend(&ret, "\n    ");
    }
  }

  if (shape_size == 1 ){
      for (int l=0; l < size; l++){
	// Weights and Bias are either generated in floating point or integer
	if (ftp) {
	  float tmpf = array[l];
	  strings::StrAppend(&ret, std::to_string(tmpf), ",");
	}
	else {
	  int tmpq = Q*(array[l]);
	  strings::StrAppend(&ret, std::to_string(tmpq), ",");
	}
	
      }
  }

  if (shape_size == 2){
    /* This part might need comment because it's a bit tricky.
       Data are not stored in the easy way to read them.
       We want them to stored in
    */
    for (int i = 0; i < shape[1]; i++){
      for(int j=0; j < last_out; j++){
        for(int k=0; k < last_w*last_h; k++){
	  // Weights and Bias are either generated in floating point or integer
	  if (ftp){
	    float tmpf = array[i+j*shape[1]+k*last_out*shape[1]];
	    strings::StrAppend(&ret, std::to_string(tmpf), ",");
	  }
	  else {
	    int tmpq = Q*(array[i+j*shape[1]+k*last_out*shape[1]]);
	    strings::StrAppend(&ret, std::to_string(tmpq), ",");
	  }
	}
	strings::StrAppend(&ret, "\n    ");
      }
      strings::StrAppend(&ret, "\n    "); 
    }
    strings::StrAppend(&ret, "\n    "); 
  }
  return ret;
}


string GAP8Tensor::GAP8SummarizeValue(int64 max_entries, int last_out, int last_w, int last_h, bool ftp) const {
  const int64 num_elts = NumElements();
  size_t limit = std::min(max_entries, num_elts);
  if ((limit > 0) && (buf() == nullptr)) {
    return strings::StrCat("uninitialized Tensor of ", num_elts,
                           " elements of type ", dtype());
  }
  const char* data = limit > 0 ? tensor_data().data() : nullptr;
  switch (dtype()) {
    case DT_HALF:
      return strings::StrCat("GAP8 unsupported deta type: " , dtype());
      break;
    case DT_FLOAT:

      return GAP8SummarizeArray<float>(limit, num_elts, shape(), data, last_out, last_w,last_h,ftp);
      break;
    case DT_DOUBLE:
      return strings::StrCat("GAP8 unsupported deta type: " , dtype());
      break;
    case DT_INT32:
      return GAP8SummarizeArray<int32>(limit, num_elts, shape(), data, last_out, last_w,last_h,ftp);
      break;
    case DT_UINT8:
    case DT_QUINT8:
      return GAP8SummarizeArray<uint8>(limit, num_elts, shape(), data, last_out, last_w,last_h,ftp);
      break;
    case DT_UINT16:
    case DT_QUINT16:
      return GAP8SummarizeArray<uint16>(limit, num_elts, shape(), data, last_out, last_w,last_h,ftp);
      break;
    case DT_INT16:
    case DT_QINT16:
      return GAP8SummarizeArray<int16>(limit, num_elts, shape(), data, last_out, last_w,last_h,ftp);
      break;
    case DT_INT8:
    case DT_QINT8:
      return GAP8SummarizeArray<int8>(limit, num_elts, shape(), data, last_out, last_w,last_h,ftp);
      break;
    case DT_INT64:
      return GAP8SummarizeArray<int64>(limit, num_elts, shape(), data, last_out, last_w,last_h,ftp);
      break;
    case DT_BOOL:
      // TODO(tucker): Is it better to emit "True False..."?  This
      // will emit "1 0..." which is more compact.
      return GAP8SummarizeArray<bool>(limit, num_elts, shape(), data, last_out, last_w,last_h,ftp);
      break;
    default: {
      // All irregular cases
      string ret;
      // TODO(irving): Don't call flat every time around this
      // loop.
      for (size_t i = 0; i < limit; ++i) {
        if (i > 0) strings::StrAppend(&ret, " ");
        switch (dtype()) {
          case DT_STRING:
            strings::StrAppend(&ret, str_util::CEscape(flat<string>()(i)));
            break;
          default:
            // TODO(zhifengc, josh11b): Pretty-print other types (bool,
            // complex64, quantized).
            strings::StrAppend(&ret, "?");
        }
      }
      if (max_entries < num_elts) strings::StrAppend(&ret, "...");
      return ret;
    }
  }
}
