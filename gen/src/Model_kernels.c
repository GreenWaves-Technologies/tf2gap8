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
#include "CNN_Generators.h"
#include "param_layer_struct.h"
#include "param_layers.h"

void CnnModel(unsigned int L1Memory)
{

  int i;
  char filename[128];
  char numker[32];

  SetInlineMode(ALWAYS_INLINE);
  SetSymbolDynamics();
  SetUsedFilesNames(0,1 , "CNN_BasicKernels.h");
  SetGeneratedFilesNames("CnnKernelsInit.c", "CnnKernelsInit.h", "CnnKernels.c", "CnnKernels.h");
  SetL1MemorySize(L1Memory);

  LoadCNNLibrary();
  CNN_LoadHWCEKernelLibrary();

  char pool_param=0;
  for (i=0;i<NB_CONV;i++)
  {
    strcpy(filename,"ConvLayer");
    sprintf(numker,"%d",i+1);
    strcat(filename,numker);
    printf("%s\n",filename);
    CNN_LargeConvolutionPoolReLU_Hor(filename, 2, 2, 2, 2, 0, 0, 0, 0, convLayers[i].nb_if, convLayers[i].nb_of, convLayers[i].win, convLayers[i].hin, convLayers[i].kernel_width, 1, 0, 0, 2, 2, 0 ,convLayers[i].relu ,convLayers[i].max_pool);
  }

  for (i=0;i<NB_DENSE;i++) {
    strcpy(filename,"Dense");
    sprintf(numker,"%d",i+1);
    strcat(filename,numker);
    printf("%s\n",filename);
    // output is on "int" format to allow different norm value for bias and output. instanciates the KerLinearRelu_fp_fp_fpd
    CNN_LinearLayerReLU(filename, 2, 2, 2, 4, 0, 0, 0, 0, denseLayers[i].win*denseLayers[i].hin*denseLayers[i].nb_if, denseLayers[i].nb_of, 0);
  }

}

int main(int argc, char **argv)
{
    if (TilerParseOptions(argc, argv))
    {
		printf("Failed to initialize or incorrect output directory.\n");
        return -1;
	}
	CnnModel(51200);
	GenerateTilingCode();
	return 0;
}

