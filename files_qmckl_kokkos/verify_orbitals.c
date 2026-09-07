#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "qmckl.h"

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.e-6;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <trexio_file>\n", argv[0]);
        return 1;
    }

    printf("--- QMCkl Kokkos Verification Harness ---\n");

    /* Initialize Context with Error Checking */
    qmckl_context ctx = qmckl_context_create();
    if (ctx == 0) {
        printf("CRITICAL ERROR: Failed to create QMCkl context (Kokkos Initialization aborted).\n");
        printf("Hint: Check your OpenMP thread binding environment variables.\n");
        return 1;
    }
    
    /* Load Data and Translate Error Codes if Failed */
    qmckl_exit_code rc = qmckl_trexio_read(ctx, argv[1], strlen(argv[1]));
    if (rc != QMCKL_SUCCESS) {
        printf("CRITICAL ERROR: Failed to read %s\n", argv[1]);
        printf("  -> Return Code : %d\n", (int)rc);
        printf("  -> Message     : %s\n", qmckl_string_of_error(rc));
        return 1;
    }

    int64_t point_num = 10000;
    double* coord = (double*) malloc(3 * point_num * sizeof(double));
    srand(42); 
    for(int i = 0; i < 3 * point_num; ++i) {
        coord[i] = ((double)rand() / RAND_MAX) * 4.0 - 2.0;
    }
    
    rc = qmckl_set_point(ctx, 'N', point_num, coord, 3 * point_num);
    if (rc != QMCKL_SUCCESS) {
        printf("CRITICAL ERROR: Failed to set evaluation points.\n");
        return 1;
    }

    int64_t mo_num;
    qmckl_get_mo_basis_mo_num(ctx, &mo_num);
    double* mo_value = (double*) malloc(point_num * mo_num * sizeof(double));

    double t1 = get_time();
    rc = qmckl_get_mo_basis_mo_value(ctx, mo_value, point_num * mo_num);
    double t2 = get_time();

    if (rc != QMCKL_SUCCESS) {
        printf("CRITICAL ERROR: MO Evaluation Failed.\n");
        return 1;
    }

    double checksum = 0.0;
    for(int i = 0; i < point_num * mo_num; ++i) {
        checksum += fabs(mo_value[i]);
    }

    printf("==================================================\n");
    printf(" Target File  : %s\n", argv[1]);
    printf(" Points (N)   : %ld\n", point_num);
    printf(" Orbitals (M) : %ld\n", mo_num);
    printf(" Time (sec)   : %f\n", t2 - t1);
    printf(" L1 Checksum  : %.10f\n", checksum);
    printf("==================================================\n");

    free(coord);
    free(mo_value);
    qmckl_context_destroy(ctx);
    return 0;
}
