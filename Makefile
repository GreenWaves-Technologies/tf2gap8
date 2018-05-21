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

ifndef GAP_TF2GAP_PATH
    $(error run 'source sourceme.sh' in gap_sdk first)
endif

LN?= ln -s
CLONE?=git clone
CHECKOUT?=git checkout
CP?=cp -f
FIND?=find
TOUCH?=touch
CD?=cd
MKDIR?=mkdir -p
RM?=rm -f

BAZEL=~/bin/bazel

INSTALL_PATH?=$(GAP_TF2GAP_PATH)/build
INSTALL_CODE=$(INSTALL_PATH)/code
INSTALL_BIN=$(INSTALL_PATH)/bin

GEN_DIR=$(GAP_TF2GAP_PATH)/examples/common
CODE_DIR=tf2gap8/code
EXAMPLES_DIR=$(GAP_TF2GAP_PATH)/examples

BUILD_TMP=$(INSTALL_PATH)/build_tmp
TF2GAP8=tf2gap8
# path to tensorflow dir
TENSORFLOW_BUILD=$(HOME)/tensorflow
# path from tensorflow to repo
TF2GAP8_FROM_BUILD=../tf2gap8
TF2GAP8_BUILD=$(TENSORFLOW_BUILD)/$(TF2GAP8)
TENSORFLOW_REPO=https://github.com/tensorflow/tensorflow.git
TENSORFLOW_VER=v1.4.0-rc1

TENSORFLOW_BUILT=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0rc1-cp35-cp35m-linux_x86_64.whl

BAZEL_INSTALLER=https://github.com/bazelbuild/bazel/releases/download/0.5.4/bazel-0.5.4-installer-linux-x86_64.sh

TENSORFLOW_MODS=$(shell find tensorflow -type f -print)

# tensorflow configure
export PYTHON_BIN_PATH=/usr/bin/python3
export USE_DEFAULT_PYTHON_LIB_PATH=1
export TF_NEED_MKL=0
export CC_OPT_FLAGS="-march=native"
export TF_NEED_JEMALLOC=1
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_ENABLE_XLA=0
export TF_NEED_VERBS=0
export TF_NEED_OPENCL=0
export TF_NEED_CUDA=0
export TF_NEED_S3=0
export TF_NEED_GDR=0
export TF_NEED_VERBS=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_MPI=0

.PHONY: prepare tensorflow_bins check_bazel bazel freezegraph transformgraph tf2gap install install-tools install-lib

all: prepare freezegraph transformgraph tf2gap

prepare: $(BUILD_TMP) bazel $(TENSORFLOW_BUILD) $(BUILD_TMP)/.tf_r12checkedout $(BUILD_TMP)/.python_virtenv $(BUILD_TMP)/.tf_configured \
	$(BUILD_TMP)/.tf2gap8_linked $(BUILD_TMP)/.tf_linked

$(BUILD_TMP): $(INSTALL_PATH)
	$(MKDIR) $@

# Clone tensorflow repo
$(TENSORFLOW_BUILD):
	$(RM) $(BUILD_TMP)/.tf_configured
	$(RM) $(BUILD_TMP)/.tf_r12checkedout
	$(RM) $(BUILD_TMP)/.tf2gap8_linked
	$(RM) $(BUILD_TMP)/.tf_linked
	$(CLONE) $(TENSORFLOW_REPO) $(TENSORFLOW_BUILD)

# Setup links between GWT source and TF
$(BUILD_TMP)/.tf_r12checkedout:
	($(CD) $(TENSORFLOW_BUILD); $(CHECKOUT) $(TENSORFLOW_VER))
	$(TOUCH) $(BUILD_TMP)/.tf_r12checkedout

$(BUILD_TMP)/.tf2gap8_linked: $(BUILD_TMP)/.tf_r12checkedout
	$(LN) -r -f -t $(TENSORFLOW_BUILD) $(TF2GAP8)
	$(TOUCH) $(BUILD_TMP)/.tf2gap8_linked

$(BUILD_TMP)/.tf_linked: $(BUILD_TMP)/.tf_r12checkedout
	$(FIND) tensorflow -type f -exec $(LN) -f '$(CURDIR)/{}' '$(TENSORFLOW_BUILD)/{}' \;
	$(TOUCH) $(BUILD_TMP)/.tf_linked

# sets up python environment and installs tensorflow
$(BUILD_TMP)/.python_env:
	$(RM) $(BUILD_TMP)/.python_virtenv
	$(RM) -r $(BUILD_TMP)/virtualenv
	sudo apt-get -y install python3-numpy python3-dev python3-pip python3-wheel python3-scipy python-virtualenv
	$(TOUCH) $(BUILD_TMP)/.python_env

$(INSTALL_PATH):
	$(MKDIR) $@

$(INSTALL_PATH)/virtualenv: $(BUILD_TMP)/.python_env | $(INSTALL_PATH)
	virtualenv --system-site-packages -p python3 $@

$(BUILD_TMP)/.python_virtenv: $(INSTALL_PATH)/virtualenv
	( . $(INSTALL_PATH)/virtualenv/bin/activate ; \
	sudo pip3 install --upgrade backports.weakref numpy ; \
	sudo pip3 install --upgrade $(TENSORFLOW_BUILT) )
	$(TOUCH) $(BUILD_TMP)/.python_virtenv

$(BUILD_TMP)/.tf_configured:
	( . $(INSTALL_PATH)/virtualenv/bin/activate ; $(CD) $(TENSORFLOW_BUILD); ./configure )
	$(TOUCH) $(BUILD_TMP)/.tf_configured

freezegraph:
	( . $(INSTALL_PATH)/virtualenv/bin/activate ; $(CD) $(TENSORFLOW_BUILD); bazel build tensorflow/python/tools:freeze_graph )

transformgraph:
	( . $(INSTALL_PATH)/virtualenv/bin/activate ; $(CD) $(TENSORFLOW_BUILD); bazel build tensorflow/tools/graph_transforms:transform_graph )

tf2gap:
	( . $(INSTALL_PATH)/virtualenv/bin/activate ; $(CD) $(TENSORFLOW_BUILD); bazel build tf2gap8:tf2gap8 )

# Install bazel
# two steps necessary here otherwise bazel never stops installing since bin/bazel is a link
bazel: check_bazel $(BUILD_TMP)/.bazel_ready

check_bazel:
	if [ -e $(BAZEL) ] ; then \
	    $(TOUCH) $(BUILD_TMP)/.bazel_ready ; \
	else \
	    $(RM) $(BUILD_TMP)/.bazel_ready ; \
	fi

$(BUILD_TMP)/.bazel_ready: $(BUILD_TMP)/.prepare_bazel $(BUILD_TMP)/install_bazel
	( chmod +x $(BUILD_TMP)/install_bazel ; $(BUILD_TMP)/install_bazel --user )
	$(TOUCH) $(BUILD_TMP)/.bazel_ready

$(BUILD_TMP)/.prepare_bazel:
	sudo apt-get -y install pkg-config zip g++ zlib1g-dev unzip python
	$(TOUCH) $(BUILD_TMP)/.prepare_bazel

$(BUILD_TMP)/install_bazel:
	wget -O $@ $(BAZEL_INSTALLER)

install: all install-tools install-lib

install-tools: $(INSTALL_BIN)/freeze_graph $(INSTALL_BIN)/transform_graph $(INSTALL_BIN)/tf2gap8

$(INSTALL_BIN):
	$(MKDIR) $@

$(INSTALL_BIN)/freeze_graph: freezegraph $(TENSORFLOW_BUILD)/bazel-bin/tensorflow/python/tools/freeze_graph $(INSTALL_BIN)
	$(CP) -r $(TENSORFLOW_BUILD)/bazel-bin/tensorflow/python/tools/freeze_graph* $(INSTALL_BIN)
	$(CP) -r $(TENSORFLOW_BUILD)/tensorflow/python/tools/freeze_graph.py $(INSTALL_BIN)
	ln -fs $(realpath $(INSTALL_BIN))/freeze_graph.py $(realpath $(INSTALL_BIN))/freeze_graph.runfiles/org_tensorflow/tensorflow/python/tools/freeze_graph.py
	chmod +w $(realpath $(INSTALL_BIN))/freeze_graph
	patch $(realpath $(INSTALL_BIN))/freeze_graph $(GAP_TF2GAP_PATH)/tf2gap8/freeze_graph.patch
	chmod -w $(realpath $(INSTALL_BIN))/freeze_graph


$(INSTALL_BIN)/transform_graph: transformgraph $(TENSORFLOW_BUILD)/bazel-bin/tensorflow/tools/graph_transforms/transform_graph $(INSTALL_BIN)
	$(CP) $(TENSORFLOW_BUILD)/bazel-bin/tensorflow/tools/graph_transforms/transform_graph $(INSTALL_BIN)

$(INSTALL_BIN)/tf2gap8: tf2gap $(TENSORFLOW_BUILD)/bazel-bin/tf2gap8/tf2gap8 $(INSTALL_BIN)
	$(CP) $(TENSORFLOW_BUILD)/bazel-bin/tf2gap8/tf2gap8 $(INSTALL_BIN)

define make_dir =
$(1):
	$(MKDIR) $$@
endef

define sync_files =
$(3)/$(1): $(2)/$(1)
	$(CP) $$< $$@
endef

# Not used in version 1.4
CODE_FILES=c_00_v2.txt c_01_v2.txt c_03_v2.txt c_03_v3.txt

CODE_PATHS=$(addprefix $(INSTALL_CODE)/, $(CODE_FILES))

install-code: $(INSTALL_CODE) $(CODE_PATHS)

$(eval $(call make_dir,$(INSTALL_CODE)))
$(foreach file, $(CODE_FILES), $(eval $(call sync_files,$(file),$(CODE_DIR),$(INSTALL_CODE))))

LIB_FILES=$(INSTALL_BIN)/freeze_graph.runfiles/org_tensorflow/tensorflow/libtensorflow_framework.so
install-lib: $(LIB_FILES)
	$(CP) $(LIB_FILES) $(INSTALL_BIN)

clean:
	$(RM) -r $(INSTALL_PATH)
