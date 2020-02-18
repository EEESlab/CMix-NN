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
 * Title:        arm_nn_mat_mult_kernel_reordered_u4_int16_u8_icn.c
 * Description:  Matrix-Multiplication function for
 *               u4 x int16_t convolution with reordered columns.
 *               Output is then quantized to u8 using icn
 *               config.folding technique.
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
 
#include "arm_cmixnn.h"

  /**
   * @brief Matrix-Multiplication function for u4 x int16_t convolution with reordered columns.
   *        Output is then quantized to u8 using icn config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       z_a         A operand offset
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t
*arm_nn_mat_mult_kernel_reordered_u4_int16_u8_icn(const uint8_t * pA,
                            const int16_t * pInBuffer,
                            const uint16_t ch_im_out,
                            const uint16_t numCol_A,
                            const int32_t * bias,
                            uint8_t * pOut,
                            const uint8_t z_a,
                            const uint8_t z_out,
                            const int32_t *m_zero,
                            const int8_t *n_zero)
{

#if defined (ARM_MATH_DSP)
    /* set up the second output pointers */
    uint8_t *pOut2 = pOut + ch_im_out;
    int     i;
    const int16_t *pB = pInBuffer;
    const int16_t *pB2 = pB + numCol_A;

    /* Negative N_ZERO Normalization */
    int8_t n_zero1;
    int8_t n_zero2;

    int16_t VzA[2] = {z_a,z_a};
    const int16_t *pzA = VzA;
    int32_t inzA = *__SIMD32(pzA);

    /* Pre-compute z_a offset over the inputs */
    int32_t z_a_offset  = 0;
    int32_t z_a_offset2 = 0;

    for (i = 0; i < numCol_A; i += 2) {
        int32_t inB1 = *__SIMD32(pB)++;
        int32_t inB2 = *__SIMD32(pB2)++;
        z_a_offset = __SMLAD(inzA, inB1, z_a_offset);
        z_a_offset2 = __SMLAD(inzA, inB2, z_a_offset2);
    }

    /* Leftover column */
    if (numCol_A & 0x1)
    {
        int16_t inB1 = *pB;
        int16_t inB2 = *pB2;
        z_a_offset += inB1*z_a;
        z_a_offset2 += inB2*z_a;
    }

    /* this loop over rows in A */
    for (i = 0; i < ch_im_out; i += 2)
    {
        /* setup pointers for B */
        pB = pInBuffer;
        pB2 = pB + numCol_A;

        /* align the second pointer for A */
        const uint8_t *pA2 = pA + (numCol_A>>1); // config.wt_data_t: u4 (4Cols per-Byte)

        int32_t     sum =  bias[i] - z_a_offset;
        int32_t     sum2 = bias[i] - z_a_offset2;
        int32_t     sum3 = bias[i + 1] - z_a_offset;
        int32_t     sum4 = bias[i + 1] - z_a_offset2;

        uint16_t  colCnt = numCol_A >> 3; // config.wt_data_t: u4 (8x uint4_t)

        /* accumulate over the vector */
        while (colCnt)
        {
            int32_t inA11, inA12, inA21, inA22;
            int32_t inA13, inA14, inA23, inA24;

            int32_t inB1 = *__SIMD32(pB)++;
            int32_t inB2 = *__SIMD32(pB2)++;

            pA = (uint8_t *) read_and_pad_reordered_u4((void *)pA, &inA11, &inA12, &inA13, &inA14);
            pA2 = (uint8_t *) read_and_pad_reordered_u4((void *)pA2, &inA21, &inA22, &inA23, &inA24);

            sum = __SMLAD(inA11, inB1, sum);
            sum2 = __SMLAD(inA11, inB2, sum2);
            sum3 = __SMLAD(inA21, inB1, sum3);
            sum4 = __SMLAD(inA21, inB2, sum4);

            inB1 = *__SIMD32(pB)++;
            inB2 = *__SIMD32(pB2)++;

            sum = __SMLAD(inA12, inB1, sum);
            sum2 = __SMLAD(inA12, inB2, sum2);
            sum3 = __SMLAD(inA22, inB1, sum3);
            sum4 = __SMLAD(inA22, inB2, sum4);

            inB1 = *__SIMD32(pB)++;
            inB2 = *__SIMD32(pB2)++;

            sum = __SMLAD(inA13, inB1, sum);
            sum2 = __SMLAD(inA13, inB2, sum2);
            sum3 = __SMLAD(inA23, inB1, sum3);
            sum4 = __SMLAD(inA23, inB2, sum4);

            inB1 = *__SIMD32(pB)++;
            inB2 = *__SIMD32(pB2)++;

            sum = __SMLAD(inA14, inB1, sum);
            sum2 = __SMLAD(inA14, inB2, sum2);
            sum3 = __SMLAD(inA24, inB1, sum3);
            sum4 = __SMLAD(inA24, inB2, sum4);

            colCnt--;
        } /* while over colCnt */

#if 0
        colCnt = numCol_A & 0x7;; // config.wt_data_t: u4 (8x uint4_t)

        int wt_per_byte = 2;
        while (colCnt)
        {
            uint8_t inB1 = (uint8_t) *pB++;
            uint8_t inA1;
            switch(wt_per_byte)
            {
                case 2:
                    inA1 = (uint8_t) __USAT(*pA, 4);
                    break;
                 case 1:
                    inA1 = (uint8_t) __USAT(__ROR(*pA, 4), 4);
                    pA++;
                    break;
            }
            inA1 -= z_a;
            sum += inA1 * inB1;
            colCnt--;
        }
#endif

        /* Normalize by ICN (u8 output) */
        __n_zero_negative_normalization(n_zero[i],&n_zero1,&n_zero2);
        sum  = ((__HI_SMULL(sum << n_zero1 ,m_zero[i])) >> n_zero2) + z_out;
        __n_zero_negative_normalization(n_zero[i],&n_zero1,&n_zero2);
        sum2  = ((__HI_SMULL(sum2 << n_zero1 ,m_zero[i])) >> n_zero2) + z_out;
        __n_zero_negative_normalization(n_zero[i+1],&n_zero1,&n_zero2);
        sum3  = ((__HI_SMULL(sum3 << n_zero1 ,m_zero[i+1])) >> n_zero2) + z_out;
        __n_zero_negative_normalization(n_zero[i+1],&n_zero1,&n_zero2);
        sum4  = ((__HI_SMULL(sum4 << n_zero1 ,m_zero[i+1])) >> n_zero2) + z_out;

        /* Store Outputs (u8 output) */
        *pOut++ = (uint8_t) __USAT(sum, 8);
        *pOut++ = (uint8_t) __USAT(sum3, 8);
        *pOut2++ = (uint8_t) __USAT(sum2, 8);
        *pOut2++ = (uint8_t) __USAT(sum4, 8);

        /* skip the row computed with A2 */
        pA += numCol_A>>1; // config.wt_data_t: u4 (2cols per-Bytes)
    } /* for over ch_im_out */
    
    pOut += ch_im_out;
#else
    #error "Cortex-M0 and Cortex-M3 not supported"
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
#endif /* ARM_MATH_DSP */

    /* return the new output pointer with offset */
    return pOut;
}
