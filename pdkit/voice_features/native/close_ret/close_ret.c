/* Close returns code by M. Little (c) 2006 */
/* Modified for Python ctypes interface */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Real variable type */
#define  REAL        double

/* Export symbol for shared library */
#if defined(_WIN32) || defined(_WIN64)
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif


/* Create embedded version of given sequence */
static void embedSeries
(
   unsigned long embedDims,      /* Number of dimensions to embed */
   unsigned long embedDelay,     /* The embedding delay */
   unsigned long embedElements,  /* Number of embedded points in embedded sequence */
   REAL          *x,             /* Input sequence */
   REAL          *y              /* (populated) Embedded output sequence */
)
{
   unsigned int i, d, inputDelay;

   for (d = 0; d < embedDims; d ++)
   {
      inputDelay = (embedDims - d - 1) * embedDelay;
      for (i = 0; i < embedElements; i ++)
      {
         y[i * embedDims + d] = x[i + inputDelay];
      }
   }
}


/* Search for first close returns in the embedded sequence */
static void findCloseReturns
(
   REAL           *x,               /* Embedded input sequence */
   REAL           eta,              /* Close return distance */
   unsigned long  embedElements,    /* Number of embedded points */
   unsigned long  embedDims,        /* Number of embedding dimensions */
   unsigned long  *closeRets        /* Close return time histogram */
)
{
   REAL  eta2 = eta * eta;
   REAL  diff, dist2;
   unsigned long  i, j, d, timeDiff, etaFlag;

   for (i = 0; i < embedElements; i ++)
   {
      closeRets[i] = 0;
   }

   for (i = 0; i < embedElements; i ++)
   {
      j = i + 1;
      etaFlag = 0;
      while ((j < embedElements) && !etaFlag)
      {
         dist2 = 0.0f;
         for (d = 0; d < embedDims; d ++)
         {
            diff   = x[i * embedDims + d] - x[j * embedDims + d];
            dist2 += diff * diff;
         }

         if (dist2 > eta2)
         {
            etaFlag = 1;
         }

         j ++;
      }

      etaFlag = 0;
      while ((j < embedElements) && !etaFlag)
      {
         dist2 = 0.0f;
         for (d = 0; d < embedDims; d ++)
         {
            diff   = x[i * embedDims + d] - x[j * embedDims + d];
            dist2 += diff * diff;
         }

         if (dist2 <= eta2)
         {
            timeDiff = j - i;
            closeRets[timeDiff] ++;
            etaFlag = 1;
         }

         j ++;
      }
   }
}


/*
 * Main entry point for Python ctypes interface
 */
EXPORT long close_ret(
    const double *input,
    unsigned long N,
    unsigned long m,
    unsigned long tau,
    double eta,
    double *output
)
{
    unsigned long i;
    unsigned long embedElements;
    REAL *embedSequence;
    unsigned long *closeRets;
    
    /* Validate inputs */
    if (input == NULL || output == NULL) {
        return -1;
    }
    if (N == 0 || m == 0 || tau == 0) {
        return -1;
    }
    
    /* Calculate embedded space size */
    if (N < (m - 1) * tau + 1) {
        return -1;  /* Signal too short for embedding */
    }
    embedElements = N - ((m - 1) * tau);
    
    /* Allocate embedded sequence */
    embedSequence = (REAL *)malloc(embedElements * m * sizeof(REAL));
    if (embedSequence == NULL) {
        return -1;
    }
    
    /* Allocate close returns histogram (using unsigned long internally) */
    closeRets = (unsigned long *)calloc(embedElements, sizeof(unsigned long));
    if (closeRets == NULL) {
        free(embedSequence);
        return -1;
    }
    
    /* Create embedded version - cast away const for embedSeries */
    REAL *inputCopy = (REAL *)malloc(N * sizeof(REAL));
    if (inputCopy == NULL) {
        free(embedSequence);
        free(closeRets);
        return -1;
    }
    memcpy(inputCopy, input, N * sizeof(REAL));
    
    /* Perform embedding */
    embedSeries(m, tau, embedElements, inputCopy, embedSequence);
    
    /* Find close returns */
    findCloseReturns(embedSequence, eta, embedElements, m, closeRets);
    
    /* Copy results to output (convert unsigned long to double) */
    for (i = 0; i < embedElements; i++) {
        output[i] = (double)closeRets[i];
    }
    
    /* Cleanup */
    free(embedSequence);
    free(closeRets);
    free(inputCopy);
    
    return (long)embedElements;
}
