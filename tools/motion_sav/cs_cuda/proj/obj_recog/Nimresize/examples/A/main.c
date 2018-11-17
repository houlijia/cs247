/*
 * main.c
 *
 * Code generation for function 'main'
 *
 */

/*************************************************************************/
/* This automatically generated example C main file shows how to call    */
/* entry-point functions that MATLAB Coder generated. You must customize */
/* this file for your application. Do not modify this file directly.     */
/* Instead, make a copy of this file, modify it, and integrate it into   */
/* your development environment.                                         */
/*                                                                       */
/* This file initializes entry-point function arguments to a default     */
/* size and value before calling the entry-point functions. It does      */
/* not store or use any values returned from the entry-point functions.  */
/* If necessary, it does pre-allocate memory for returned values.        */
/* You can use this file as a starting point for a main function that    */
/* you can deploy in your application.                                   */
/*                                                                       */
/* After you copy the file, and before you deploy it, you must make the  */
/* following changes:                                                    */
/* * For variable-size function arguments, change the example sizes to   */
/* the sizes that your application requires.                             */
/* * Change the example values of function arguments to the values that  */
/* your application requires.                                            */
/* * If the entry-point functions return values, store these values or   */
/* otherwise use them as required by your application.                   */
/*                                                                       */
/*************************************************************************/
/* Include files */
#include "rt_nonfinite.h"
#include "Nimresize.h"
#include "main.h"
#include "Nimresize_terminate.h"
#include "Nimresize_emxAPI.h"
#include "Nimresize_initialize.h"
#include <stdio.h>

/* Function Declarations */
static void argInit_1x2_real_T(double result[2]);
static void argInit_480x640x3_uint8_T(unsigned char result[921600]);
static double argInit_real_T(void);
static unsigned char argInit_uint8_T(void);
static void main_Nimresize(void);

/* Function Definitions */
static void argInit_1x2_real_T(double result[2])
{
  int b_j1;

  /* Loop over the array to initialize each element. */
  for (b_j1 = 0; b_j1 < 2; b_j1++) {
    /* Set the value of the array element.
       Change this value to the value that the application requires. */
    result[b_j1] = argInit_real_T();
  }
}

static void argInit_480x640x3_uint8_T(unsigned char result[921600])
{
  int b_j0;
  int b_j1;
  int j2;

  /* Loop over the array to initialize each element. */
  for (b_j0 = 0; b_j0 < 480; b_j0++) {
    for (b_j1 = 0; b_j1 < 640; b_j1++) {
      for (j2 = 0; j2 < 3; j2++) {
        /* Set the value of the array element.
           Change this value to the value that the application requires. */
        result[(b_j0 + 480 * b_j1) + 307200 * j2] = argInit_uint8_T();
      }
    }
  }
}

static double argInit_real_T(void)
{
  return 0.0;
}

static unsigned char argInit_uint8_T(void)
{
  return 0;
}

static void main_Nimresize(void)
{
  emxArray_uint8_T *rout;
  static unsigned char uv0[921600];
  double dv3[2];
  emxInitArray_uint8_T(&rout, 3);

  /* Initialize function 'Nimresize' input arguments. */
  /* Initialize function input argument 'A'. */
  /* Initialize function input argument 'm'. */
  /* Call the entry-point 'Nimresize'. */
  argInit_480x640x3_uint8_T(uv0);
  argInit_1x2_real_T(dv3);
  Nimresize(uv0, dv3, rout);
  emxDestroyArray_uint8_T(rout);
}

int main(int argc, const char * const argv[])
{
  (void)argc;
  (void)argv;

  /* Initialize the application.
     You do not need to do this more than one time. */
  Nimresize_initialize();

  /* Invoke the entry-point functions.
     You can call entry-point functions multiple times. */
  main_Nimresize();

  /* Terminate the application.
     You do not need to do this more than one time. */
  Nimresize_terminate();
  return 0;
}

/* End of code generation (main.c) */
