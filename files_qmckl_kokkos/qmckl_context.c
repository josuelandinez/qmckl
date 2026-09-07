#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_STDINT_H
#include <stdint.h>
#elif HAVE_INTTYPES_H
#include <inttypes.h>
#endif

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include "qmckl.h"
#include "qmckl_context_private_type.h"
#include "qmckl_memory_private_func.h"

qmckl_context qmckl_context_check(const qmckl_context context) {

  if (context == QMCKL_NULL_CONTEXT)
    return QMCKL_NULL_CONTEXT;

  const qmckl_context_struct* const ctx = (qmckl_context_struct*) context;

  /* Try to access memory */
  if (ctx->tag != VALID_TAG) {
      return QMCKL_NULL_CONTEXT;
  }

  return context;
}

qmckl_exit_code
qmckl_context_touch(const qmckl_context context)
{

  if (qmckl_context_check(context) == QMCKL_NULL_CONTEXT) {
    return qmckl_failwith( context,
                           QMCKL_INVALID_CONTEXT,
                           "qmckl_context_touch",
                           NULL);
  }

  qmckl_context_struct* const ctx = (qmckl_context_struct*) context;

  ctx->date += 1UL;
  ctx->point.date = ctx-> date;
  ctx->electron.walker.point.date = ctx-> date;
  return QMCKL_SUCCESS;
}

qmckl_context qmckl_context_create() {

       qmckl_context_struct* const ctx =
	 (qmckl_context_struct*) malloc (sizeof(qmckl_context_struct));

       if (ctx == NULL) {
	 return QMCKL_NULL_CONTEXT;
       }

#ifdef HAVE_KOKKOS
       /* Secretly initialize the hardware backend if Kokkos is present */
       extern void qmckl_kokkos_initialize();
       qmckl_kokkos_initialize();
#endif

       /* Set all pointers and values to NULL */
       {
	 memset(ctx, 0, sizeof(qmckl_context_struct));
       }

       /* Initialize lock */
       {
	 pthread_mutexattr_t attr;
	 int rc;

	 rc = pthread_mutexattr_init(&attr);
	 assert (rc == 0);

     #ifdef PTHREAD_MUTEX_RECURSIVE
	 (void) pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
     #endif

	 rc = pthread_mutex_init ( &(ctx->mutex), &attr);
	 assert (rc == 0);

	 (void) pthread_mutexattr_destroy(&attr);
       }

       /* Initialize data */
       {
	 ctx->tag = VALID_TAG;

	 const qmckl_context context = (qmckl_context) ctx;
	 assert ( qmckl_context_check(context) != QMCKL_NULL_CONTEXT );

	 qmckl_exit_code rc;

	 ctx->numprec.precision = QMCKL_DEFAULT_PRECISION;
	 ctx->numprec.range = QMCKL_DEFAULT_RANGE;

     #ifdef HAVE_OPENMP
	 ctx->num_threads = omp_get_max_threads();
     #else
	 ctx->num_threads = 1;
     #endif

	 rc = qmckl_init_point(context);
	 assert (rc == QMCKL_SUCCESS);

	 rc = qmckl_init_electron(context);
	 assert (rc == QMCKL_SUCCESS);

	 rc = qmckl_init_nucleus(context);
	 assert (rc == QMCKL_SUCCESS);

	 rc = qmckl_init_ao_basis(context);
	 assert (rc == QMCKL_SUCCESS);

	 rc = qmckl_init_mo_basis(context);
	 assert (rc == QMCKL_SUCCESS);

	 /*
	 rc = qmckl_init_determinant(context);
	 assert (rc == QMCKL_SUCCESS);
	 */

	 rc = qmckl_init_jastrow_champ(context);
	 assert (rc == QMCKL_SUCCESS);
       }

       /* Allocate qmckl_memory_struct */
       {
	 const size_t size = 128L;
	 qmckl_memory_info_struct * new_array = calloc(size, sizeof(qmckl_memory_info_struct));
	 if (new_array == NULL) {
	   free(ctx);
	   return QMCKL_NULL_CONTEXT;
	 }
	 memset( &(new_array[0]), 0, size * sizeof(qmckl_memory_info_struct) );

	 ctx->memory.element = new_array;
	 ctx->memory.array_size = size;
	 ctx->memory.n_allocated = (size_t) 0;
       }

       return (qmckl_context) ctx;
     }

void qmckl_lock(qmckl_context context) {
  if (context == QMCKL_NULL_CONTEXT)
    return ;
  qmckl_context_struct* const ctx = (qmckl_context_struct*) context;
  errno = 0;
  int rc = pthread_mutex_lock( &(ctx->mutex) );
  if (rc != 0) {
    fprintf(stderr, "DEBUG qmckl_lock:%s\n", strerror(rc) );
    fflush(stderr);
  }
  assert (rc == 0);
  ctx->lock_count += 1;
}

void qmckl_unlock(const qmckl_context context) {
  qmckl_context_struct* const ctx = (qmckl_context_struct*) context;
  int rc = pthread_mutex_unlock( &(ctx->mutex) );
  if (rc != 0) {
    fprintf(stderr, "DEBUG qmckl_unlock:%s\n", strerror(rc) );
    fflush(stderr);
  }
  assert (rc == 0);
  ctx->lock_count -= 1;
}

qmckl_context qmckl_context_copy(const qmckl_context context) {
  qmckl_exit_code rc;

  const qmckl_context checked_context = qmckl_context_check(context);

  if (checked_context == QMCKL_NULL_CONTEXT) {
    return QMCKL_NULL_CONTEXT;
  }

  qmckl_lock(context);
  {

    const qmckl_context_struct* const old_ctx =
      (qmckl_context_struct*) checked_context;

    /* Create a new context using the standard creation function */
    qmckl_context new_context = qmckl_context_create();
    if (new_context == QMCKL_NULL_CONTEXT) {
      qmckl_unlock(context);
      return QMCKL_NULL_CONTEXT;
    }

    qmckl_context_struct* const new_ctx =
      (qmckl_context_struct*) new_context;

    /* Copy scalar values and structs that don't contain pointers */
    new_ctx->date = old_ctx->date;
    new_ctx->numprec = old_ctx->numprec;
    new_ctx->error = old_ctx->error;
    new_ctx->num_threads = old_ctx->num_threads;

    /* Deep copy point structure using dedicated function */
    rc = qmckl_copy_point(new_context, &(old_ctx->point), &(new_ctx->point));
    if (rc != QMCKL_SUCCESS) {
      qmckl_context_destroy(new_context);
      qmckl_unlock(context);
      return QMCKL_NULL_CONTEXT;
    }

    /* Deep copy single_point structure using dedicated function */
    rc = qmckl_copy_jastrow_champ_single(new_context, &(old_ctx->single_point), &(new_ctx->single_point));
    if (rc != QMCKL_SUCCESS) {
      qmckl_context_destroy(new_context);
      qmckl_unlock(context);
      return QMCKL_NULL_CONTEXT;
    }

    /* Deep copy nucleus structure using dedicated function */
    rc = qmckl_copy_nucleus(new_context, &(old_ctx->nucleus), &(new_ctx->nucleus));
    if (rc != QMCKL_SUCCESS) {
      qmckl_context_destroy(new_context);
      qmckl_unlock(context);
      return QMCKL_NULL_CONTEXT;
    }

    /* Deep copy electron structure using dedicated function */
    rc = qmckl_copy_electron(new_context, &(old_ctx->electron), &(new_ctx->electron));
    if (rc != QMCKL_SUCCESS) {
      qmckl_context_destroy(new_context);
      qmckl_unlock(context);
      return QMCKL_NULL_CONTEXT;
    }

    /* Deep copy ao_basis structure using dedicated function */
    rc = qmckl_copy_ao_basis(new_context, &(old_ctx->ao_basis), &(new_ctx->ao_basis));
    if (rc != QMCKL_SUCCESS) {
      qmckl_context_destroy(new_context);
      qmckl_unlock(context);
      return QMCKL_NULL_CONTEXT;
    }

    /* Deep copy mo_basis structure using dedicated function */
    rc = qmckl_copy_mo_basis(new_context, &(old_ctx->mo_basis), &(new_ctx->mo_basis));
    if (rc != QMCKL_SUCCESS) {
      qmckl_context_destroy(new_context);
      qmckl_unlock(context);
      return QMCKL_NULL_CONTEXT;
    }

    /* Deep copy jastrow_champ structure using dedicated function */
    rc = qmckl_copy_jastrow_champ(new_context, &(old_ctx->jastrow_champ), &(new_ctx->jastrow_champ));
    if (rc != QMCKL_SUCCESS) {
      qmckl_context_destroy(new_context);
      qmckl_unlock(context);
      return QMCKL_NULL_CONTEXT;
    }

    /* Deep copy forces structure using dedicated function */
    rc = qmckl_copy_forces(new_context, &(old_ctx->forces), &(new_ctx->forces));
    if (rc != QMCKL_SUCCESS) {
      qmckl_context_destroy(new_context);
      qmckl_unlock(context);
      return QMCKL_NULL_CONTEXT;
    }

    /* Copy qmckl_extra pointer (shallow copy - implementation specific) */
    new_ctx->qmckl_extra = old_ctx->qmckl_extra;

    qmckl_unlock(context);
    return new_context;
  }
}

qmckl_exit_code qmckl_context_get_num_threads(const qmckl_context context, int32_t* const num_threads) {
  if (qmckl_context_check(context) == QMCKL_NULL_CONTEXT)
    return QMCKL_INVALID_CONTEXT;

  if (num_threads == NULL) {
    return qmckl_failwith(context,
                          QMCKL_INVALID_ARG_2,
                          "qmckl_context_get_num_threads",
                          "num_threads is a null pointer");
  }

  const qmckl_context_struct* const ctx = (qmckl_context_struct*) context;
  *num_threads = ctx->num_threads;
  return QMCKL_SUCCESS;
}

qmckl_exit_code qmckl_context_set_num_threads(const qmckl_context context, const int32_t num_threads) {
  if (qmckl_context_check(context) == QMCKL_NULL_CONTEXT)
    return QMCKL_INVALID_CONTEXT;

  if (num_threads <= 0) {
    return qmckl_failwith(context,
                          QMCKL_INVALID_ARG_2,
                          "qmckl_context_set_num_threads",
                          "num_threads must be positive");
  }

  qmckl_context_struct* const ctx = (qmckl_context_struct*) context;
  assert (ctx != NULL);

  qmckl_lock(context);
  {
    ctx->num_threads = num_threads;
  }
  qmckl_unlock(context);

  return QMCKL_SUCCESS;
}

qmckl_exit_code qmckl_context_unset_num_threads(const qmckl_context context) {
  if (qmckl_context_check(context) == QMCKL_NULL_CONTEXT)
    return QMCKL_INVALID_CONTEXT;

  qmckl_context_struct* const ctx = (qmckl_context_struct*) context;
  assert (ctx != NULL);

  qmckl_lock(context);
  {
#ifdef HAVE_OPENMP
    ctx->num_threads = omp_get_max_threads();
#else
    ctx->num_threads = 1;
#endif
  }
  qmckl_unlock(context);

  return QMCKL_SUCCESS;
}

qmckl_exit_code
      qmckl_context_destroy (const qmckl_context context)
      {

	const qmckl_context checked_context = qmckl_context_check(context);
	if (checked_context == QMCKL_NULL_CONTEXT) return QMCKL_INVALID_CONTEXT;

	qmckl_context_struct* const ctx = (qmckl_context_struct*) context;
	assert (ctx != NULL);  /* Shouldn't be possible because the context is valid */

	qmckl_lock(context);
	{
	  /* Memory: Remove all allocated data */
	  for (size_t pos = (size_t) 0 ; pos < ctx->memory.array_size ; ++pos) {
	    if (ctx->memory.element[pos].pointer != NULL) {
	      free(ctx->memory.element[pos].pointer);
	      memset( &(ctx->memory.element[pos]), 0, sizeof(qmckl_memory_info_struct) );
	      ctx->memory.n_allocated -= 1;
	    }
	  }
	  assert (ctx->memory.n_allocated == (size_t) 0);
	  free(ctx->memory.element);
	  ctx->memory.element = NULL;
	  ctx->memory.array_size = (size_t) 0;
	}
	qmckl_unlock(context);

	ctx->tag = INVALID_TAG;

	const int rc_destroy = pthread_mutex_destroy( &(ctx->mutex) );
	if (rc_destroy != 0) {
      /* DEBUG */
	   fprintf(stderr, "qmckl_context_destroy: %s (count = %d)\n", strerror(rc_destroy), ctx->lock_count);
	   abort();
	}

#ifdef HAVE_KOKKOS
	/* Secretly shutdown the hardware backend */
	extern void qmckl_kokkos_finalize();
	qmckl_kokkos_finalize();
#endif

	free(ctx);

	return QMCKL_SUCCESS;
      }
