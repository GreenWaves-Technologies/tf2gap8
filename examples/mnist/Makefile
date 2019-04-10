# Copyright (c) 2017 GreenWaves Technologies SAS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of GreenWaves Technologies SAS nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

ifndef GAP_TF2GAP_PATH
	$(error run 'source sourceme.sh' in gap_sdk first)
endif

include $(GAP_TF2GAP_PATH)/gen/tf2gap8.mk

T2G_BUILD_DIR=tfbuild
T2G_CODE_DIR=$(T2G_BUILD_DIR)/GAP8Code
T2G_EXAMPLES_COMMON=../common

T2G_KERNEL_SRCS = $(wildcard $(T2G_CNNKERNEL_SRC)/*.c)

PULP_APP = mnist
GEN_SRCS = $(T2G_BUILD_DIR)/CnnKernels.c $(T2G_BUILD_DIR)/CnnKernelsInit.c  $(T2G_CODE_DIR)/weights_bias.c $(T2G_CODE_DIR)/network_process.c $(CNN_KERNELS_LIST)

PULP_APP_SRCS = l2_x_mnist.c $(T2G_EXAMPLES_COMMON)/main.c $(GEN_SRCS)
PULP_CFLAGS  += -O3 -Wno-unused-but-set-variable -Wno-pointer-sign #-DDUMP_LAYER
PULP_CFLAGS  += -I$(T2G_GENTILING_INC) -I$(T2G_GEN_INC_DIR)  -I$(T2G_STDTYPES) -I$(T2G_CNNSTDMODEL) -I$(T2G_BUILD_DIR) -I$(T2G_CODE_DIR) -I${T2G_CNNKERNEL_INC}

$(eval $(call tf2gap8_rules,$(CURDIR)/data,mnist.pbtxt,model.ckpt,x_inter,y_output,$(CURDIR)/$(T2G_BUILD_DIR),mnist, false))

all:: mnist $(GEN_SRCS)

clean::
	rm -rf $(T2G_BUILD_DIR)

include $(GAP_SDK_HOME)/tools/rules/pulp_rules.mk


