#ifndef MUMPS_CONSTANTS_H
#define MUMPS_CONSTANTS_H

// MUMPS control parameters
constexpr int mumps_comm_world = -987654;  // use MPI_COMM_WORLD (sequential fallback)

// MUMPS job codes
constexpr int mumps_job_init   = -1;  // initialize instance
constexpr int MUMPS_JOB_END    = -2;  // terminate instance
constexpr int MUMPS_JOB_ANALYZE = 1;  // analysis only
constexpr int MUMPS_JOB_FACTOR  = 2;  // factorization only
constexpr int MUMPS_JOB_SOLVE   = 3;  // solve only
constexpr int MUMPS_JOB_ANALYZE_FACTOR = 4;  // analysis + factorization
constexpr int MUMPS_JOB_FACTOR_SOLVE = 6;  // factorization + solve

// MUMPS icntl[0..3] defaults for host-driven parallel factorization
constexpr int MUMPS_ICNTL_ERRORS  = -1;  // suppress error messages
constexpr int MUMPS_ICNTL_DIAG    = -1;  // suppress diagnostic output
constexpr int MUMPS_ICNTL_GLOBAL  = -1;  // suppress global info
constexpr int MUMPS_ICNTL_MEMORY  =  0;  // let MUMPS manage memory
constexpr int MUMPS_ICNTL_RHS_SPARSE = 1;  // sparse right-hand side

#endif
