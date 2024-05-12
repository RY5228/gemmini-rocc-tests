
#include <stdio.h>
#include "include/gemmini.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

// batch size: 64
// before zeropad: 10x50x100x200x100x50x10
// after zeropad: 64x64x128x256x128x64x64
elem_t input_mat[64][64] row_align(1)= {0};
elem_t weights0[64][64] row_align(1)= {0};
elem_t inter_results0[64][64] row_align(1)= {0};
elem_t weights1[64][128] row_align(1)= {0};
elem_t inter_results1[64][128] row_align(1)= {0};
elem_t weights2[128][256] row_align(1)= {0};
elem_t inter_results2[64][256] row_align(1)= {0};
elem_t weights3[256][128] row_align(1)= {0};
elem_t inter_results3[64][128] row_align(1)= {0};
elem_t weights4[128][64] row_align(1)= {0};
elem_t inter_results4[64][64] row_align(1)= {0};
elem_t weights5[64][64] row_align(1)= {0};
elem_t inter_results5[64][64] row_align(1)= {0};
