/* Copyright (C) 2020 University of Bologna
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

 /******************************************************************
 * Project:      CMixNN Inference Library
 * Title:        arm_cmixnn.h
 * Description:  Public header file for CMix-NN Support Functions.
 * 
 * Target:       ARM Cortex-M cores
 *
 * Date:         10 February 2020
 * Revision:     Release v1.0.0
 *
 * Authors:      Alessandro Capotondi
 *                 <alessandro.capotondi AT unibo.it>
 *               Marco Fariselli
 *                 <marco.fariselli AT greenwaves-technologies.com>
 *               Manuele Rusci 
 *                 <manuele.rusci AT unibo.it>
 *
 ******************************************************************/

#ifndef _ARM_MIXNN_SUPPORT_H_
#define _ARM_MIXNN_SUPPORT_H_

#include "arm_math.h"

#ifdef __cplusplus
extern "C"
{
#endif

  // /**
  //  * @brief Union for SIMD access of Q31/Q15/Q7 types
  //  */
  // union arm_nnword
  // {
  //     q31_t     word;
  //                /**< Q31 type */
  //     q15_t     half_words[2];
  //                /**< Q15 type */
  //     q7_t      bytes[4];
  //                /**< Q7 type */
  // };

  /**
 * @defgroup CMixNN_convert CMixNN Data Conversion Functions
 * To be implemented.
 */

  ${CMixNNSupportAPI}

  /*
 * @brief read and expand one u8 word into 2x2xint16_t with reordering
 */
  __STATIC_INLINE void *
  read_and_pad_reordered_u8(void *source, int32_t *out1, int32_t *out2)
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
  __STATIC_INLINE void *read_and_pad_reordered_u4(void *source, int32_t *out1, int32_t *out2, int32_t *out3, int32_t *out4)
  {
    int32_t inA = *__SIMD32(source)++;
#ifndef ARM_MATH_BIG_ENDIAN
    *out1 = __UXTB16(inA & 0x000F000F);
    *out2 = __UXTB16(__ROR(inA, 4) & 0x000F000F);
    *out3 = __UXTB16(__ROR(inA, 8) & 0x000F000F);
    *out4 = __UXTB16(__ROR(inA, 12) & 0x000F000F);
#else
  *out4 = __UXTB16(inA & 0x000F000F);
  *out3 = __UXTB16(__ROR(inA, 4) & 0x000F000F);
  *out2 = __UXTB16(__ROR(inA, 8) & 0x000F000F);
  *out1 = __UXTB16(__ROR(inA, 12) & 0x000F000F);
#endif
    return source;
  }

  /**
  * @brief read and expand one u2 word into 8x2xint16_t with reordering
  */
  __STATIC_INLINE void *read_and_pad_reordered_u2(void *source, int32_t *out1, int32_t *out2, int32_t *out3, int32_t *out4,
                                                  int32_t *out5, int32_t *out6, int32_t *out7, int32_t *out8)
  {
    q31_t inA = *__SIMD32(source)++;
#ifndef ARM_MATH_BIG_ENDIAN
    *out1 = __UXTB16(inA & 0x00030003);
    *out2 = __UXTB16(__ROR(inA, 2) & 0x00030003);
    *out3 = __UXTB16(__ROR(inA, 4) & 0x00030003);
    *out4 = __UXTB16(__ROR(inA, 6) & 0x00030003);
    *out5 = __UXTB16(__ROR(inA, 8) & 0x00030003);
    *out6 = __UXTB16(__ROR(inA, 10) & 0x00030003);
    *out7 = __UXTB16(__ROR(inA, 12) & 0x00030003);
    *out8 = __UXTB16(__ROR(inA, 14) & 0x00030003);
#else
  *out8 = __UXTB16(inA & 0x00030003);
  *out7 = __UXTB16(__ROR(inA, 2) & 0x00030003);
  *out6 = __UXTB16(__ROR(inA, 4) & 0x00030003);
  *out5 = __UXTB16(__ROR(inA, 6) & 0x00030003);
  *out4 = __UXTB16(__ROR(inA, 8) & 0x00030003);
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
    asm volatile("SMULL %[lo_out], %[hi_out], %[a_operand], %[b_operand]"
                 : [lo_out] "=&r"(lo), [hi_out] "=&r"(hi)
                 : [a_operand] "r"(a), [b_operand] "r"(b));
    return hi;
  }

  __STATIC_INLINE void __n_zero_negative_normalization(int8_t n_zero, int8_t *n_zero1, int8_t *n_zero2)
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

  /* Threasholds from int16 to u4 */
  __STATIC_INLINE uint8_t __int16_to_u4(int16_t input, const int16_t *pThr)
  {
    uint8_t ret = 0;
    if (input <= pThr[7])
    {
      if (input <= pThr[3])
      {
        if (input <= pThr[1])
        {
          if (input <= pThr[0])
            ret = 0;
          else
            ret = 1;
        }
        else
        {
          if (input <= pThr[2])
            ret = 2;
          else
            ret = 3;
        }
      }
      else
      {
        if (input <= pThr[5])
        {
          if (input <= pThr[4])
            ret = 4;
          else
            ret = 5;
        }
        else
        {
          if (input <= pThr[6])
            ret = 6;
          else
            ret = 7;
        }
      }
    }
    else
    {
      if (input <= pThr[11])
      {
        if (input <= pThr[9])
        {
          if (input <= pThr[8])
            ret = 8;
          else
            ret = 9;
        }
        else
        {
          if (input <= pThr[10])
            ret = 10;
          else
            ret = 11;
        }
      }
      else
      {
        if (input <= pThr[13])
        {
          if (input <= pThr[12])
            ret = 12;
          else
            ret = 13;
        }
        else
        {
          if (input <= pThr[14])
            ret = 14;
          else
            ret = 15;
        }
      }
    }
    return ret;
  }

  /* Threasholds from int16 to u2 */
  __STATIC_INLINE uint8_t __int16_to_u2(int16_t input, const int16_t *pThr)
  {
    uint8_t ret = 0;
    if (input <= pThr[1])
    {
      if (input <= pThr[0])
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
      if (input <= pThr[2])
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

#ifdef __cplusplus
}
#endif

#endif
