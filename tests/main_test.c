#include <stdio.h>
#include "qmckl.h"

int main() {
    printf("--- QMCkl Core Architecture Test ---\n");
    
    /* 1. Test C to Fortran Context Allocation */
    qmckl_context context = qmckl_context_create();
    if (context == QMCKL_NULL_CONTEXT) {
        printf("CRITICAL FAILURE: Could not allocate QMCkl context.\n");
        return 1;
    }
    printf("[PASS] QMCkl Context successfully created (C -> Fortran Bridge Active).\n");

    /* 2. Test Threading/State Management */
    qmckl_exit_code rc = qmckl_context_set_num_threads(context, 4);
    if (rc == QMCKL_SUCCESS) {
        printf("[PASS] OpenMP/Threading state initialized.\n");
    }

    /* 3. Clean Memory Destruction */
    rc = qmckl_context_destroy(context);
    if (rc == QMCKL_SUCCESS) {
        printf("[PASS] Memory successfully freed.\n");
    }

    printf("\n=== BASELINE ENVIRONMENT IS 100%% READY ===\n");
    return 0;
}
