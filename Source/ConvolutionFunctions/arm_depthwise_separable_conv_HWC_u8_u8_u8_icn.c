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
 * Title:        arm_depthwise_separable_conv_HWC_u8_u8_u8_icn.c
 * Description:  Mixed Precision Depthwise Convolutional function
 *               that uses u8 activations, u8 weights
 *               and produce u8 output activations.
 *               Outputs are quantized using icn folding technique.
 {config.header_bottom}

#include <assert.h>

#include "arm_math.h"
#include "arm_cmixnn.h"

/**
 *  @ingroup CMIXNN
 */

/**
 * @addtogroup CMIXConv
 * @{
 */

 /**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u8 weights
   *        and produce u8 output activations. Outputs are quantized using icn folding technique.
   *
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       left_pad    padding sizes
   * @param[in]       right_pad   padding sizes
   * @param[in]       top_pad     padding sizes
   * @param[in]       bottom_pad  padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in]       z_in        input offset
   * @param[in]       z_wt        weights offset
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u8_u8_icn(const uint8_t * Im_in,
                        const uint16_t dim_im_in,
                        const uint16_t ch_im_in,
                        const uint8_t * wt,
                        const uint16_t ch_im_out,
                        const uint16_t dim_kernel,
                        const uint8_t left_padding,
                        const uint8_t right_padding,
                        const uint8_t top_padding,
                        const uint8_t bottom_padding,
                        const uint16_t stride,
                        const int32_t * bias,
                        uint8_t * Im_out,
                        const uint16_t dim_im_out,
                        const uint8_t z_in,
                        const uint8_t z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB)
{

#if defined (ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t   i_out_y, i_out_x;
    int16_t   i_ker_y, i_ker_x;
    uint8_t    *colBuffer = (uint8_t *) bufferA;
    uint8_t    *pBuffer = colBuffer;
    const int32_t *pBias = bias;
    uint8_t   *pOut = Im_out;
    uint16_t  rowCnt;
    uint16_t  row_shift;

    int16_t Vz_wt[2] = {z_wt,z_wt};
    const int32_t *pz_wt = (int32_t *)Vz_wt;
    int32_t inz_wt = *__SIMD32(pz_wt);

    int16_t Vz_in[2] = {z_in,z_in};
    const int32_t *pz_in = (int32_t *) Vz_in;
    int32_t inz_in = *__SIMD32(pz_in);

    /* Negative N_ZERO Normalization */
    int8_t n_zero1;
    int8_t n_zero2;

    /* do some checking here, basically ch_im_in == ch_im_out */
    if (ch_im_in != ch_im_out)
    {
        return ARM_MATH_SIZE_MISMATCH;
    }

    for (i_out_y = 0; i_out_y < dim_im_out; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            /* we first do im2col here */
            for (i_ker_y = i_out_y * stride - top_padding; i_ker_y < i_out_y * stride - top_padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - left_padding; i_ker_x < i_out_x * stride - left_padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        memset(pBuffer, 0, ch_im_in);
                    }
                    else
                    {
                        memcpy(pBuffer, Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            /* we will do the computation here for each channel */
            rowCnt = ch_im_out >> 2;
            row_shift = 0;
            pBias = bias;
            int32_t ch_out_id=0;

            while (rowCnt)
            {
                int32_t     sum =  *pBias++;
                int32_t     sum2 = *pBias++;
                int32_t     sum3 = *pBias++;
                int32_t     sum4 = *pBias++;

                uint16_t    colCnt = (dim_kernel * dim_kernel) >> 1;
                uint8_t     *pB = colBuffer + row_shift;
                const uint8_t *pA = wt + row_shift;

#ifndef ARM_MATH_BIG_ENDIAN

                while (colCnt)
                {
                    int32_t     inA1, inA2, inB1, inB2, opA, opB;

                    inB1 = *__SIMD32(pB);
                    pB += ch_im_in;
                    opB = *__SIMD32(pB);
                    pB += ch_im_in;
                    inB2 = __PKHTB(opB, inB1, 16);
                    inB1 = __PKHBT(inB1, opB, 16);

                    inA1 = *__SIMD32(pA);
                    pA += ch_im_in;
                    opB = *__SIMD32(pA);
                    pA += ch_im_in;
                    inA2 = __PKHTB(opB, inA1, 16);
                    inA1 = __PKHBT(inA1, opB, 16);

                    //sum
                    opA = __UXTB16(inA1);
                    opB = __UXTB16(inB1);

                    opA = __SSUB16(opA, inz_wt);
                    opB = __SSUB16(opB, inz_in);
                    sum = __SMLAD(opA, opB, sum);

                    //sum2
                    opA = __UXTB16(__ROR(inA1, 8));
                    opB = __UXTB16(__ROR(inB1, 8));

                    opA = __SSUB16(opA, inz_wt);
                    opB = __SSUB16(opB, inz_in);
                    sum2 = __SMLAD(opA, opB, sum2);

                    //sum3
                    opA = __UXTB16(inA2);
                    opB = __UXTB16(inB2);

                    opA = __SSUB16(opA, inz_wt);
                    opB = __SSUB16(opB, inz_in);
                    sum3 = __SMLAD(opA, opB, sum3);

                    //sum4
                    opA = __UXTB16(__ROR(inA2, 8));
                    opB = __UXTB16(__ROR(inB2, 8));

                    opA = __SSUB16(opA, inz_wt);
                    opB = __SSUB16(opB, inz_in);
                    sum4 = __SMLAD(opA, opB, sum4);
                    colCnt--;
                }
#else
#error Missing Big Endian Implementation
#endif /* ARM_MATH_BIG_ENDIAN */

                colCnt = (dim_kernel * dim_kernel) & 0x1;
                while (colCnt)
                {
                    union arm_nnword inA, inB;
                    inA.word = *__SIMD32(pA);
                    pA += ch_im_in;
                    inB.word = *__SIMD32(pB);
                    pB += ch_im_in;
                    sum  += (((uint8_t) inA.bytes[0])-z_wt) * (((uint8_t) inB.bytes[0])-z_in);
                    sum2 += (((uint8_t) inA.bytes[1])-z_wt) * (((uint8_t) inB.bytes[1])-z_in);
                    sum3 += (((uint8_t) inA.bytes[2])-z_wt) * (((uint8_t) inB.bytes[2])-z_in);
                    sum4 += (((uint8_t) inA.bytes[3])-z_wt) * (((uint8_t) inB.bytes[3])-z_in);
                    colCnt--;
                }

                /* Normalize by ICN (u8 output) */
                __n_zero_negative_normalization(n_zero[ch_out_id],&n_zero1,&n_zero2);
                sum  = ((__HI_SMULL(sum << n_zero1 ,m_zero[ch_out_id++])) >> n_zero2) + z_out;
                __n_zero_negative_normalization(n_zero[ch_out_id],&n_zero1,&n_zero2);
                sum2  = ((__HI_SMULL(sum2 << n_zero1 ,m_zero[ch_out_id++])) >> n_zero2) + z_out;
                __n_zero_negative_normalization(n_zero[ch_out_id],&n_zero1,&n_zero2);
                sum3  = ((__HI_SMULL(sum3 << n_zero1 ,m_zero[ch_out_id++])) >> n_zero2) + z_out;
                __n_zero_negative_normalization(n_zero[ch_out_id],&n_zero1,&n_zero2);
                sum4  = ((__HI_SMULL(sum4 << n_zero1 ,m_zero[ch_out_id++])) >> n_zero2) + z_out;

                /* Store Outputs (u8 output) */
                *pOut++ = (uint8_t) __USAT(sum, 8);
                *pOut++ = (uint8_t) __USAT(sum2, 8);
                *pOut++ = (uint8_t) __USAT(sum3, 8);
                *pOut++ = (uint8_t) __USAT(sum4, 8);

                row_shift += 4;
                rowCnt--;
            }


            rowCnt = ch_im_out & 0x3;
            while (rowCnt)
            {
                uint8_t     *pB = colBuffer + row_shift;
                int32_t     sum = *pBias++;
                uint16_t  colCnt = (dim_kernel * dim_kernel);

                const uint8_t *pA = wt + row_shift;
                row_shift += 1;

                while (colCnt)
                {
                    uint8_t B1 = *pB;
                    uint8_t A1 = (uint8_t) __USAT(*pA, 8);
                    pA += ch_im_in;
                    pB += ch_im_in;

                    A1 -= z_wt;
                    B1 -= z_in;
                    sum += A1 * B1;

                    colCnt--;
                }

                /* Normalize by ICN (u8 output) */
                __n_zero_negative_normalization(n_zero[ch_out_id],&n_zero1,&n_zero2);
                sum  = ((__HI_SMULL(sum << n_zero1 ,m_zero[ch_out_id++])) >> n_zero2) + z_out;

                /* Store Outputs (u8 output) */
                *pOut++ = (uint8_t) __USAT(sum, 8);
                rowCnt--;
            }
            /* clear counter and pointers */
            pBuffer = colBuffer;
        }
    }
#else
    #error "Cortex-M0 and Cortex-M3 not supported"
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
#endif                          /* ARM_MATH_DSP */

    /* Return to application */
    return ARM_MATH_SUCCESS;
}

/**
 * @} end of CMIXConv group
 */
