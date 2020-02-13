/*
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.
 * Modifications Copyright (C) 2019 University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nnsupportfunctions.h
 * Description:  Public header file of support functions for CMSIS NN Library
 *
 * Target Processor:  Cortex-M cores
 * 
 * Modification: Mixed-Precision INT-Q extension
 *
 * $Date:        3 September 2019
 * $Revision:    V.1.2.0
 *
 * $Authors:     Alessandro Capotondi - alessandro.capotondi@unibo.it
 *               Marco Fariselli - marco.fariselli2@unibo.it 
 *               Manuele Rusci - manuele.rusci@unibo.it
 *               
 * -------------------------------------------------------------------- */

#ifndef _ARM_NNSUPPORTFUNCTIONS_H_
#define _ARM_NNSUPPORTFUNCTIONS_H_

#include "arm_math.h"
#include "arm_common_tables.h"
//#include <cstring>

#ifdef __cplusplus
extern    "C"
{
#endif

/**
 * @brief Union for SIMD access of Q31/Q15/Q7 types
 */
union arm_nnword
{
    q31_t     word;
               /**< Q31 type */
    q15_t     half_words[2];
               /**< Q15 type */
    q7_t      bytes[4];
               /**< Q7 type */
};

/**
 * @brief Struct for specifying activation function types
 *
 */
typedef enum
{
    ARM_SIGMOID = 0,
                /**< Sigmoid activation function */
    ARM_TANH = 1,
             /**< Tanh activation function */
} arm_nn_activation_type;

/**
 * @defgroup nndata_convert Neural Network Data Conversion Functions
 *
 * Perform data type conversion in-between neural network operations
 *
 */

/**
 * @brief Converts the elements of the Q7 vector to Q15 vector without left-shift 
 * @param[in]       *pSrc points to the Q7 input vector    
 * @param[out]      *pDst points to the Q15 output vector   
 * @param[in]       blockSize length of the input vector    
 * @return none.    
 *
 */

void      arm_q7_to_q15_no_shift(const q7_t * pSrc, q15_t * pDst, uint32_t blockSize);

/**
 * @brief  Converts the elements of the Q7 vector to reordered Q15 vector without left-shift
 * @param[in]       *pSrc points to the Q7 input vector    
 * @param[out]      *pDst points to the Q15 output vector   
 * @param[in]       blockSize length of the input vector    
 * @return none.
 */
void      arm_q7_to_q15_reordered_no_shift(const q7_t * pSrc, q15_t * pDst, uint32_t blockSize);

${CMSISSupportAPI}

#if defined (ARM_MATH_DSP)

/**
 * @brief read and expand one Q7 word into two Q15 words
 */

__STATIC_FORCEINLINE void *read_and_pad(void *source, q31_t * out1, q31_t * out2)
{
        q31_t     inA = *__SIMD32(source)++;
        q31_t     inAbuf1 = __SXTB16(__ROR(inA, 8));
        q31_t     inAbuf2 = __SXTB16(inA);

#ifndef ARM_MATH_BIG_ENDIAN
        *out2 = __PKHTB(inAbuf1, inAbuf2, 16);
        *out1 = __PKHBT(inAbuf2, inAbuf1, 16);
#else
        *out1 = __PKHTB(inAbuf1, inAbuf2, 16);
        *out2 = __PKHBT(inAbuf2, inAbuf1, 16);
#endif

        return source;
}

/**
 * @brief read and expand one Q7 word into two Q15 words with reordering
 */

__STATIC_FORCEINLINE void *read_and_pad_reordered(void *source, q31_t * out1, q31_t * out2)
{
        q31_t     inA = *__SIMD32(source)++;
#ifndef ARM_MATH_BIG_ENDIAN
        *out2 = __SXTB16(__ROR(inA, 8));
        *out1 = __SXTB16(inA);
#else
        *out1 = __SXTB16(__ROR(inA, 8));
        *out2 = __SXTB16(inA);
#endif

        return source;
}

/*
 * @brief read and expand one u8 word into 2x2xint16_t with reordering
 */
__STATIC_INLINE void *read_and_pad_reordered_u8(void *source, int32_t * out1, int32_t * out2)
{
        int32_t inA = *__SIMD32(source)++;
#ifndef ARM_MATH_BIG_ENDIAN
        *out2 = __UXTB16(__ROR(inA, 8));
        *out1 = __UXTB16(inA);
#else
        *out1 = __UXTB16(__ROR(inA, 8));
        *out2 = __UXTB16(inA);
#endif

        return source;
}

/**
  * @brief read and expand one u4 word into 4x2xint16_t with reordering
  */
__STATIC_INLINE void *read_and_pad_reordered_u4(void *source, int32_t * out1, int32_t * out2, int32_t * out3, int32_t * out4)
{
        int32_t     inA = *__SIMD32(source)++;
#ifndef ARM_MATH_BIG_ENDIAN
        *out1 = __UXTB16(      inA      & 0x000F000F);
        *out2 = __UXTB16(__ROR(inA, 4)  & 0x000F000F);
        *out3 = __UXTB16(__ROR(inA, 8)  & 0x000F000F);
        *out4 = __UXTB16(__ROR(inA, 12) & 0x000F000F);
#else
        *out4 = __UXTB16(      inA      & 0x000F000F);
        *out3 = __UXTB16(__ROR(inA, 4)  & 0x000F000F);
        *out2 = __UXTB16(__ROR(inA, 8)  & 0x000F000F);
        *out1 = __UXTB16(__ROR(inA, 12) & 0x000F000F);
#endif
        return source;
}

/**
  * @brief read and expand one u2 word into 8x2xint16_t with reordering
  */
__STATIC_INLINE void *read_and_pad_reordered_u2(  void *source, int32_t * out1, int32_t * out2, int32_t * out3, int32_t * out4,
                                                    int32_t * out5, int32_t * out6, int32_t * out7, int32_t * out8)
{
        q31_t     inA = *__SIMD32(source)++;
#ifndef ARM_MATH_BIG_ENDIAN
        *out1 = __UXTB16(      inA      & 0x00030003);
        *out2 = __UXTB16(__ROR(inA, 2)  & 0x00030003);
        *out3 = __UXTB16(__ROR(inA, 4)  & 0x00030003);
        *out4 = __UXTB16(__ROR(inA, 6)  & 0x00030003);
        *out5 = __UXTB16(__ROR(inA, 8)  & 0x00030003);
        *out6 = __UXTB16(__ROR(inA, 10) & 0x00030003);
        *out7 = __UXTB16(__ROR(inA, 12) & 0x00030003);
        *out8 = __UXTB16(__ROR(inA, 14) & 0x00030003);
#else
        *out8 = __UXTB16(      inA      & 0x00030003);
        *out7 = __UXTB16(__ROR(inA, 2)  & 0x00030003);
        *out6 = __UXTB16(__ROR(inA, 4)  & 0x00030003);
        *out5 = __UXTB16(__ROR(inA, 6)  & 0x00030003);
        *out4 = __UXTB16(__ROR(inA, 8)  & 0x00030003);
        *out3 = __UXTB16(__ROR(inA, 10) & 0x00030003);
        *out2 = __UXTB16(__ROR(inA, 12) & 0x00030003);
        *out1 = __UXTB16(__ROR(inA, 14) & 0x00030003);
#endif
        return source;
}

__STATIC_INLINE int32_t __HI_SMULL(int32_t a, int32_t b)
{
  int hi = 0;
  int lo = 0;
  asm volatile ("SMULL %[lo_out], %[hi_out], %[a_operand], %[b_operand]"
    : [lo_out] "=&r" (lo), [hi_out] "=&r" (hi)
    : [a_operand] "r" (a), [b_operand] "r" (b)
  );
  return hi;
}


__STATIC_INLINE void __n_zero_negative_normalization(int8_t n_zero, int8_t *n_zero1, int8_t*n_zero2)
{
  if (n_zero > 0)
  {
    *n_zero1 = 0;
    *n_zero2 = n_zero;
  }
  else
  {
    *n_zero1 = -n_zero;
    *n_zero2 = 0;
  }
}

#endif

/* Threasholds from int16 to u4 */
__STATIC_INLINE uint8_t __int16_to_u4(int16_t input, const int16_t * pThr) {
  uint8_t ret = 0;
  if(input <= pThr[7] )
  {
    if( input <= pThr[3])
    {
      if( input <= pThr[1])
      {
        if( input <= pThr[0])
          ret = 0;
        else
          ret = 1;
      }
      else
      {
        if( input <= pThr[2])
          ret = 2;
        else
          ret = 3;
      }
    }
    else
    {
      if( input <= pThr[5])
      {
        if( input <= pThr[4])
          ret = 4;
        else
          ret = 5;
      }
      else
      {
        if( input <= pThr[6])
          ret = 6;
        else
          ret = 7;
      }
    }
  }
  else
  {
    if( input <= pThr[11])
    {
      if( input <= pThr[9])
      {
        if( input <= pThr[8])
          ret = 8;
        else
          ret = 9;
      }
      else
      {
        if( input <= pThr[10])
          ret = 10;
        else
          ret =11;
      }
    }
    else
    {
      if( input <= pThr[13])
      {
        if( input <= pThr[12])
          ret = 12;
        else
          ret = 13;
      }
      else
      {
        if( input <= pThr[14])
          ret = 14;
        else
          ret = 15;
      }
    }
  }
  return ret;
}

/* Threasholds from int16 to u2 */
__STATIC_INLINE uint8_t __int16_to_u2(int16_t input, const int16_t * pThr){
  uint8_t ret = 0;
  if( input <= pThr[1])
    {
    if( input <= pThr[0])
        {
      ret = 0;
    }
        else
        {
      ret = 1;
    }
  }
    else
    {
    if( input <= pThr[2])
        {
      ret = 2;
    }
        else
        {
      ret = 3;
    }
  }
  return ret;
}

/**
 * @defgroup NNBasicMath Basic Math Functions for Neural Network Computation
 *
 * Basic Math Functions for Neural Network Computation
 *
 */

/**
 * @brief           Q7 vector multiplication with variable output shifts
 * @param[in]       *pSrcA        pointer to the first input vector
 * @param[in]       *pSrcB        pointer to the second input vector
 * @param[out]      *pDst         pointer to the output vector
 * @param[in]       out_shift     amount of right-shift for output
 * @param[in]       blockSize     number of samples in each vector
 * @return none.
 *
 * <b>Scaling and Overflow Behavior:</b>
 * \par
 * The function uses saturating arithmetic.
 * Results outside of the allowable Q15 range [0x8000 0x7FFF] will be saturated.
 */

void arm_nn_mult_q15(
  q15_t * pSrcA,
  q15_t * pSrcB,
  q15_t * pDst,
  const uint16_t out_shift,
  uint32_t blockSize);
  
/**
 * @brief           Q7 vector multiplication with variable output shifts
 * @param[in]       *pSrcA        pointer to the first input vector
 * @param[in]       *pSrcB        pointer to the second input vector
 * @param[out]      *pDst         pointer to the output vector
 * @param[in]       out_shift     amount of right-shift for output
 * @param[in]       blockSize     number of samples in each vector
 * @return none.
 *
 * <b>Scaling and Overflow Behavior:</b>
 * \par
 * The function uses saturating arithmetic.
 * Results outside of the allowable Q7 range [0x80 0x7F] will be saturated.
 */

void arm_nn_mult_q7(
  q7_t * pSrcA,
  q7_t * pSrcB,
  q7_t * pDst,
  const uint16_t out_shift,
  uint32_t blockSize);
 
/**
 * @brief defition to adding rouding offset
 */
#ifndef ARM_NN_TRUNCATE
    #define NN_ROUND(out_shift) ( 0x1 << (out_shift - 1) )
#else
    #define NN_ROUND(out_shift) 0
#endif

#ifdef __cplusplus
}
#endif

#endif
