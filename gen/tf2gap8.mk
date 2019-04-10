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

# Macro for T2G_TF2GAP8 code generation

MKDIR?=mkdir -p
RM?=rm -f

ifndef GAP_SDK_HOME
    $(error run 'source sourceme.sh' in gap_sdk first)
endif

T2G_SDK_HOME=$(GAP_SDK_HOME)

T2G_AT_ROOT=$(realpath $(T2G_SDK_HOME)/tools/autotiler)
#T2G_GENTILING=$(T2G_AT_ROOT)/GenTiling
T2G_GENTILING=$(T2G_AT_ROOT)/lib
T2G_GENTILING_INC=$(T2G_AT_ROOT)/include
T2G_STDTYPES=$(T2G_AT_ROOT)/StdTypes
#T2G_CNNSTDMODEL=$(T2G_AT_ROOT)/autotiler_generator/CnnStdModel
T2G_CNNSTDMODEL=$(T2G_AT_ROOT)/generators/CNN/generator/src
T2G_CNNSTDMODEL_INC=$(T2G_AT_ROOT)/generators/CNN/generator/include
T2G_CNNGENERATOR=$(T2G_CNNSTDMODEL)/CNN_Generators.c
T2G_LIBTILE=$(T2G_GENTILING)/libtile.a
T2G_CNNKERNEL_SRC= $(T2G_AT_ROOT)/generators/CNN/kernels/src
T2G_CNNKERNEL_INC = $(T2G_AT_ROOT)/generators/CNN/kernels/include
T2G_TF2GAP8_ROOT=$(T2G_SDK_HOME)/tf2gap8/build
# Python environment is in tf2gap/virtualenv
T2G_PYTHONENV=$(T2G_TF2GAP8_ROOT)/virtualenv
T2G_ACTIVATE=$(T2G_PYTHONENV)/bin/activate

T2G_TOOLS_DIR=$(T2G_TF2GAP8_ROOT)/bin
T2G_GEN_DIR=$(GAP_TF2GAP_PATH)/gen
T2G_GEN_SRC_DIR=$(T2G_GEN_DIR)/src
T2G_GEN_INC_DIR=$(T2G_GEN_DIR)/inc

T2G_MODEL_KERNELS=$(T2G_GEN_SRC_DIR)/Model_kernels.c

T2G_FREEZE=freeze_graph
T2G_FREEZE_PATH=$(T2G_TOOLS_DIR)/$(T2G_FREEZE)
T2G_TRANSFORM=transform_graph
T2G_TRANSFORM_PATH=$(T2G_TOOLS_DIR)/$(T2G_TRANSFORM)

T2G_TF2GAP8=tf2gap8
T2G_TF2GAP8_PATH=$(T2G_TOOLS_DIR)/$(T2G_TF2GAP8)
T2G_TF2GAP8_CODE_DIR=$(T2G_TF2GAP8_ROOT)/code

T2G_TRANSFORMS=strip_unused_nodes remove_nodes(op=Identity) fuse_conv2d_add_relu_maxpool \
                            fuse_conv2d_add_relu \
                            fuse_conv2d_add_maxpool \
                            fuse_GAP8_conv2d_maxpool \
                            fuse_reshape_matmul_add_relu_softmax \
                            fuse_reshape_matmul_add_softmax \
                            fuse_reshape_matmul_add_relu \
                            fuse_reshape_matmul_add \
                            fuse_matmul_add_relu \
                            fuse_matmul_add \
                            fuse_depthwiseconv2d_add

T2G_GEN_FILES=CnnKernels.c CnnKernels.h CnnKernelsInit.c CnnKernelsInit.h
CNN_KERNELS_LIST := $(wildcard $(T2G_CNNKERNEL_SRC)/*.c) 

# $1 = Directory with input TensorFlow graph
# $2 = Input graph filename e.g. mnist.pbtxt
# $3 = Input graph checkpoint filename prefix e.g. model.ckpt
# $4 = Input node name e.g. x_inter
# $5 = Output node name e.g. y_output
# $6 = Build directory e.g. $(CURDIR)/tfbuild
# $7 = Make phony target for generation e.g. mnist

define tf2gap8_rules =
T2G_$(7)_INPUT_DIR=$1
T2G_$(7)_INPUT_GRAPH_NAME=$2
T2G_$(7)_INPUT_CHECKPOINT_NAME?=$3
T2G_$(7)_INPUT_NODE=$4
T2G_$(7)_OUTPUT_NODE=$5
T2G_$(7)_OUTPUT_DIR=$6

T2G_$(7)_INPUT_GRAPH_PATH=$1/$2
T2G_$(7)_INPUT_CHECKPOINT_PATH=$1/$3

T2G_$(7)_FROZEN_GRAPH_PATH=$6/$(7)_frozen.pb
T2G_$(7)_OPTIMIZED_GRAPH_PATH=$6/$(7)_optimized.pb
T2G_$(7)_CODE_GEN_PATH=$6/GAP8Code

T2G_$(7)_T2G_GEN_FILES=$(addprefix $6/,$(T2G_GEN_FILES))

.PHONY: $(7)

$(7): $$(T2G_$(7)_T2G_GEN_FILES)

$$(T2G_$(7)_OUTPUT_DIR):
	$(MKDIR) $$@

$$(T2G_$(7)_FROZEN_GRAPH_PATH): $$(T2G_$(7)_INPUT_GRAPH_PATH) | $$(T2G_$(7)_OUTPUT_DIR)
	@echo "\n--- T2G_FREEZE GRAPH ---"
	( . $(T2G_ACTIVATE) ; $(T2G_FREEZE_PATH) --input_graph="$$(T2G_$(7)_INPUT_GRAPH_PATH)" --input_checkpoint="$$(T2G_$(7)_INPUT_CHECKPOINT_PATH)" \
	  --output_graph="$$(T2G_$(7)_FROZEN_GRAPH_PATH)" --output_node_names="$$(T2G_$(7)_OUTPUT_NODE)" )

$$(T2G_$(7)_OPTIMIZED_GRAPH_PATH): $$(T2G_$(7)_FROZEN_GRAPH_PATH)
	@echo "\n--- T2G_TRANSFORM GRAPH ---"
	rm -f $$(T2G_$(7)_OUTPUT_DIR)/.run_$(7)_tf2gap8
	( . $(T2G_ACTIVATE) ; $(T2G_TRANSFORM_PATH) --in_graph="$$(T2G_$(7)_FROZEN_GRAPH_PATH)" --out_graph="$$(T2G_$(7)_OPTIMIZED_GRAPH_PATH)" \
	  --inputs="$$(T2G_$(7)_INPUT_NODE)" --outputs="$$(T2G_$(7)_OUTPUT_NODE)" --transforms="$(T2G_TRANSFORMS)" )

$$(T2G_$(7)_OUTPUT_DIR)/.run_$(7)_tf2gap8: $$(T2G_$(7)_OPTIMIZED_GRAPH_PATH)
	@echo "\n--- GENERATE GRAPH CODE DESCRIPTION ---"
	( . $(T2G_ACTIVATE) ; $(T2G_TF2GAP8_PATH) "$$(notdir $$(T2G_$(7)_OPTIMIZED_GRAPH_PATH))" "$$(T2G_$(7)_OUTPUT_DIR)" "$(T2G_TF2GAP8_ROOT)" $(8))
	touch $$(T2G_$(7)_OUTPUT_DIR)/.run_$(7)_tf2gap8

$$(T2G_$(7)_OUTPUT_DIR)/$(7)_tilegen: $$(T2G_$(7)_OUTPUT_DIR)/.run_$(7)_tf2gap8
	@echo "\n--- COMPILE CODE GENERATOR ---"
	$(CC) -g -o $$@ -I$(T2G_GENTILING) -I$(T2G_GENTILING_INC) -I$(T2G_GEN_INC_DIR) -I$(T2G_STDTYPES) -I$(T2G_CNNSTDMODEL) -I$(T2G_CNNSTDMODEL_INC) -I$$(T2G_$(7)_CODE_GEN_PATH) $(T2G_MODEL_KERNELS) $(T2G_CNNGENERATOR) $(T2G_LIBTILE)

$$(T2G_$(7)_T2G_GEN_FILES) : $$(T2G_$(7)_OUTPUT_DIR)/$(7)_tilegen
	@echo "\n--- RUN GENERATOR ---"
	( cd $$(T2G_$(7)_OUTPUT_DIR) ; ./$(7)_tilegen )

endef

# clean:
# 	$(RM) -r $(OUTPUT_DIR)
