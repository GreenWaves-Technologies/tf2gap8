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

#include <stdint.h>
#include <stdio.h>
#include <string.h>
// Import AutoTiler lib
#include "AutoTilerLib.h"
// Import CNN generators
#include "CNN_Generator.h"
#include "param_layer_struct.h"
#include "param_layers.h"

void CnnModel(unsigned int L1Memory)

{


  int i;
  char filename[128];
  char numker[32];

  SetInlineMode(ALWAYS_INLINE);
  SetSymbolDynamics();
  SetUsedFilesNames("KernelLibStdTypes.h",1 , "CNN_BasicKernels.h");
  SetGeneratedFilesNames("CnnKernelsInit.c", "CnnKernelsInit.h", "CnnKernels.c", "CnnKernels.h");

  SetL1MemorySize(L1Memory);

  CNN_LoadSoftwareKernelLibrary();
  CNN_LoadHWCEKernelLibrary();
  // cifar config


  char pool_param=0;
  for (i=0;i<NB_CONV;i++) {
    strcpy(filename,"ConvLayer");
    sprintf(numker,"%d",i+1);
    strcat(filename,numker);
    printf("%s\n",filename);
    // set pool_param parameter : AVG pool is not supported in theis release
    if (convLayers[i].max_pool && convLayers[i].relu) pool_param=1;
    // infer max pooling
    if (convLayers[i].max_pool && (!convLayers[i].relu)) pool_param=3;
    if ((!convLayers[i].max_pool)) printf(" only conv layer with pooling is supported in this version\n");
   CNN_TiledConvNxNReLUPool2x2_SW_fp(filename, convLayers[i].kernel_width, convLayers[i].nb_if, convLayers[i].nb_of, convLayers[i].win, convLayers[i].hin, pool_param);

  }

  for (i=0;i<NB_DENSE;i++) {
    strcpy(filename,"Dense");
    sprintf(numker,"%d",i+1);
    strcat(filename,numker);
    printf("%s\n",filename);
    CNN_TiledLinearLayer(filename, denseLayers[i].nb_if, denseLayers[i].win, denseLayers[i].hin, denseLayers[i].nb_of, 1, denseLayers[i].relu, 0 );
  }



}

int main(int argc, char **argv)

{
  if (TilerParseOptions(argc, argv)) {
		printf("Failed to initialize or incorrect output directory.\n"); return 0;
	}
	CnnModel(51200);
	GenerateTilingCode();
	return 0;
}

