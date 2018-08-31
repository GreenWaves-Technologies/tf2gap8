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

#ifndef CCN_PULP
  #include <stdio.h>
  #include <stdint.h>
#endif

#ifdef STACK_SIZE
#undef STACK_SIZE
#endif
#define STACK_SIZE 1024

#define MOUNT           1
#define UNMOUNT         0
#define CID             0

#ifndef NULL
  #define NULL ((void *) 0)
#endif

int finished = 0;

#include "network_process.h"

RT_L2_DATA short int l2_big0[BUF0_SIZE];
RT_L2_DATA short int l2_big1[BUF1_SIZE];


static rt_perf_t *cluster_perf;
extern Kernel_T AllKernels[];

static void cluster_main()
{
  printf ("cluster master start\n");

  printf("CNN  running on %d cores\n", rt_nb_pe());

  network_process();
}

static void end_of_app()
{
  finished = 1;
  printf ("End of CNN process\n");
}

int main()
{
    char error =0;
    printf("start main\n");

    rt_event_sched_t sched;
    rt_event_sched_init(&sched);
    if (rt_event_alloc(&sched, 4)) return -1;

    rt_cluster_mount(MOUNT, CID, 0, NULL);

    void *stacks = rt_alloc(RT_ALLOC_CL_DATA, STACK_SIZE*rt_nb_pe());
    if (stacks == NULL) return -1;

    L1_Memory       = rt_alloc(RT_ALLOC_CL_DATA, _L1_Memory_SIZE) ;
    if(L1_Memory == NULL) {
        printf("WorkingArea alloc error\n");
        return -1;
    }

    cluster_perf = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(rt_perf_t));
    if (cluster_perf == NULL) {
        printf("cluster perf allocation failed");
        return -1;
    }

    rt_cluster_call(NULL, CID, cluster_main, NULL, stacks, STACK_SIZE, STACK_SIZE, rt_nb_pe(), 0);

    rt_cluster_mount(UNMOUNT, CID, 0, NULL);

    int i;
    int sum = 0;
    char *s;

    printf("\n");
    int  max=0x80000000;
    unsigned char idx=0;

    for(i=0;i<CLast_NFEAT; i++){
        printf(" feat %d: %d  \n ", i, l2_big0[i]);
        sum += l2_big0[i];
        if (l2_big0[i]>max) {max=l2_big0[i];idx=i;}
    }

    printf("found %d\n",idx);

#ifdef CHECK_CHECKSUM
    if((sum == CHECKSUM)) {
        printf("CHECKSUM OK!!! (%d)\n", sum);
    }
    else {
        printf("CHECKSUM NOK!!! (%d instead of %d)\n\n", sum, CHECKSUM);
        error = 1;
    }
#endif
    return error;
}
