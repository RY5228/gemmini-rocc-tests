

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "parameters9.h"

// #define verbose(layer_num,old_C,filter,C) printf("layer %d: operand %d %d filter %d %d result %d %d\n", layer_num, LEN(old_C),LEN(old_C[0]),LEN(filter),LEN(filter[0]),LEN(C),LEN(C[0]));
// 
// static void tiled_matmul_compare(size_t DIM_I, size_t DIM_J, size_t DIM_K,
//         elem_t A[DIM_I][DIM_K], elem_t B[DIM_K][DIM_J], acc_t D[DIM_I][DIM_J],
//         elem_t C[DIM_I][DIM_J], int no_bias, int act, int shift, int relu6_shift,
//         enum tiled_matmul_type_t tiled_matmul_type,
//         bool compare, char * layer_name)
// {
//     if (compare)
//         printf("%s: gemmini\n", layer_name);
//     tiled_matmul_option(DIM_I, DIM_J, DIM_K,
//         A, B, D, C, no_bias, act, shift, relu6_shift,
//         tiled_matmul_type);
// 
//     if (compare) {
//         printf("%s: CPU\n", layer_name);
//         elem_t gold[DIM_I][DIM_J];
//         tiled_matmul_option(DIM_I, DIM_J, DIM_K,
//             A, B, D, gold, no_bias, act, shift, relu6_shift,
//             CPU);
// 
//         printf("%s: comparing\n", layer_name);
//         for (size_t i = 0; i < DIM_I; i++) {
//             for (size_t j = 0; j < DIM_J; j++) {
//                 if (C[i][j] != gold[i][j]) {
//                     printf("Layer calculated incorrectly: %s\n", layer_name);
//                     exit(1);
//                 }
//             }
//         }
//     }
// }

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type;
    if (argc < 2) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "os") == 0) {
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "-h") == 0) {
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(0);
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }


    bool check;
    if (argc < 3) {
        check = false;
    } else if (strcmp(argv[2], "check") == 0) {
        check = true;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }


    uint64_t cycles[6]={0};
    uint64_t start,end;


    /* matmul number: 0 */

    start = read_cycles();
    tiled_matmul_nn_auto(64, 64, 64,    // dimensions
        input_mat, weights0, NULL, inter_results0,      // addresses
        RELU, 0, false,              // no_bias, act, shift, r6_shift
        tiled_matmul_type, check, "layer_0");
    // verbose(0,input_mat,weights0,inter_results0)
    /* end of matmul number: 0 */

    end = read_cycles();
    cycles[0] = end-start;


    /* matmul number: 1 */

    start = read_cycles();
    tiled_matmul_nn_auto(64, 128, 64,    // dimensions
        inter_results0, weights1, NULL, inter_results1,      // addresses
        RELU, 0, false,              // no_bias, act, shift, r6_shift
        tiled_matmul_type, check, "layer_1");
    // verbose(1,inter_results0,weights1,inter_results1)
    /* end of matmul number: 1 */

    end = read_cycles();
    cycles[1] = end-start;


    /* matmul number: 2 */

    start = read_cycles();
    tiled_matmul_nn_auto(64, 256, 128,    // dimensions
        inter_results1, weights2, NULL, inter_results2,      // addresses
        RELU, 0, false,              // no_bias, act, shift, r6_shift
        tiled_matmul_type, check, "layer_2");
    // verbose(2,inter_results1,weights2,inter_results2)
    /* end of matmul number: 2 */

    end = read_cycles();
    cycles[2] = end-start;


    /* matmul number: 3 */

    start = read_cycles();
    tiled_matmul_nn_auto(64, 128, 256,    // dimensions
        inter_results2, weights3, NULL, inter_results3,      // addresses
        RELU, 0, false,              // no_bias, act, shift, r6_shift
        tiled_matmul_type, check, "layer_3");
    // verbose(3,inter_results2,weights3,inter_results3)
    /* end of matmul number: 3 */

    end = read_cycles();
    cycles[3] = end-start;


    /* matmul number: 4 */

    start = read_cycles();
    tiled_matmul_nn_auto(64, 64, 128,    // dimensions
        inter_results3, weights4, NULL, inter_results4,      // addresses
        RELU, 0, false,              // no_bias, act, shift, r6_shift
        tiled_matmul_type, check, "layer_4");
    // verbose(4,inter_results3,weights4,inter_results4)
    /* end of matmul number: 4 */

    end = read_cycles();
    cycles[4] = end-start;


    /* matmul number: 5 */

    start = read_cycles();
    tiled_matmul_nn_auto(64, 64, 64,    // dimensions
        inter_results4, weights5, NULL, inter_results5,      // addresses
        RELU, 0, false,              // no_bias, act, shift, r6_shift
        tiled_matmul_type, check, "layer_5");
    // verbose(5,inter_results4,weights5,inter_results5)
    /* end of matmul number: 5 */

    end = read_cycles();
    cycles[5] = end-start;

    uint64_t overall_cycles = 0;
    for(int cyc = 0; cyc < 6 ; cyc++){
        overall_cycles += cycles[cyc];
        printf("Cycles taken in layer %d: %llu\n", cyc,cycles[cyc]);
    }
    printf("Overall cycles taken: %llu\n",overall_cycles);


    return 0;
}

