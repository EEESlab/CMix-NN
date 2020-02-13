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
 * Title:        arm_convolve_HWC_u4_u2_u4_icn.c
 * Description:  Mixed Precision Convolutional function that uses u4
 *               activations, u4 weights and produce u2
 *               output activations. Outputs are quantized using icn
 *               folding technique.
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
   * @brief Mixed Precision Convolution icn (in: u4, out: u2, wt: u4)
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
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status
arm_convolve_HWC_u4_u2_u4_icn(const uint8_t *Im_in,
                    const uint16_t dim_im_in,
                    const uint16_t ch_im_in,
                    const uint8_t *wt,
                    const uint16_t ch_im_out,
                    const uint16_t dim_kernel,
                    const uint8_t left_padding,
                    const uint8_t right_padding,
                    const uint8_t top_padding,
                    const uint8_t bottom_padding,
                    const uint16_t stride,
                    const int32_t *bias,
                    uint8_t *Im_out,
                    const uint16_t dim_im_out,
                    const uint8_t z_in,
                    const uint8_t z_wt,
                    const uint8_t z_out,
                    const int32_t *m_zero,
                    const int8_t *n_zero,
                    int16_t * bufferA,
                    uint8_t *bufferB)
{

#if defined(ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t i_out_y, i_out_x, i_ker_y, i_ker_x;
    int16_t *pBuffer = bufferA;
    uint8_t *pOut = Im_out;

    if (ch_im_in % 8 != 0 || ch_im_out % 16 != 0)
    {
        /* check if the input dimension meets the constraints */
        return ARM_MATH_SIZE_MISMATCH;
    }

    /*
     *  Here we split the entire matrix into three regions depending on the padding situation
     *    Top: i_out_y from 0 to padding - 1
     * Middle: i_out_y from padding to dim_im_out-padding-1
     * Bottom: i_out_y from dim_im_out-padding to dim_im_out-1
     */

    /* top part */
    for (i_out_y = 0; i_out_y < top_padding; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - top_padding; i_ker_y < i_out_y * stride - top_padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - left_padding; i_ker_x < i_out_x * stride - left_padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        memset(pBuffer, 0, sizeof(int16_t) * ch_im_in);
                    }
                    else
                    {
                        arm_u4_to_int16_reordered(
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 1),
                                                pBuffer,
                                                ch_im_in,
                                                z_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut = arm_nn_mat_mult_kernel_reordered_u4_int16_u2_icn(wt,
                                                bufferA,
                                                ch_im_out,
                                                ch_im_in * dim_kernel * dim_kernel,
                                                bias,
                                                pOut,
                                                z_wt,
                                                z_out,
                                                m_zero,
                                                n_zero);
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    /* middle part, here we also divide the x into left, mid and right */
    for (; i_out_y < dim_im_out - bottom_padding; i_out_y++)
    {

        /* left part */
        for (i_out_x = 0; i_out_x < left_padding; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - top_padding; i_ker_y < i_out_y * stride - top_padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - left_padding; i_ker_x < i_out_x * stride - left_padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        memset(pBuffer, 0, sizeof(int16_t) * ch_im_in);
                    }
                    else
                    {
                        arm_u4_to_int16_reordered(
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 1),
                                                pBuffer,
                                                ch_im_in,
                                                z_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut = arm_nn_mat_mult_kernel_reordered_u4_int16_u2_icn(wt,
                                                bufferA,
                                                ch_im_out,
                                                ch_im_in * dim_kernel * dim_kernel,
                                                bias,
                                                pOut,
                                                z_wt,
                                                z_out,
                                                m_zero,
                                                n_zero);
                /* counter reset */
                pBuffer = bufferA;
            }
        }

        /* mid part */
        for (; i_out_x < dim_im_out - right_padding; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - top_padding; i_ker_y < i_out_y * stride - top_padding + dim_kernel; i_ker_y++)
            {
                arm_u4_to_int16_reordered(
                                                Im_in + (((i_ker_y * dim_im_in + i_out_x * stride - top_padding) * ch_im_in) >> 1),
                                                pBuffer,
                                                ch_im_in * dim_kernel,
                                                z_in);
                pBuffer += ch_im_in * dim_kernel;
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut = arm_nn_mat_mult_kernel_reordered_u4_int16_u2_icn(wt,
                                                bufferA,
                                                ch_im_out,
                                                ch_im_in * dim_kernel * dim_kernel,
                                                bias,
                                                pOut,
                                                z_wt,
                                                z_out,
                                                m_zero,
                                                n_zero);
                /* counter reset */
                pBuffer = bufferA;
            }
        }

        /* right part */
        for (; i_out_x < dim_im_out; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - top_padding; i_ker_y < i_out_y * stride - top_padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - left_padding; i_ker_x < i_out_x * stride - left_padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        memset(pBuffer, 0, sizeof(int16_t) * ch_im_in);
                    }
                    else
                    {
                        arm_u4_to_int16_reordered(
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 1),
                                                pBuffer,
                                                ch_im_in,
                                                z_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut = arm_nn_mat_mult_kernel_reordered_u4_int16_u2_icn(wt,
                                                bufferA,
                                                ch_im_out,
                                                ch_im_in * dim_kernel * dim_kernel,
                                                bias,
                                                pOut,
                                                z_wt,
                                                z_out,
                                                m_zero,
                                                n_zero);
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    for (; i_out_y < dim_im_out; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - top_padding; i_ker_y < i_out_y * stride - top_padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - left_padding; i_ker_x < i_out_x * stride - left_padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        memset(pBuffer, 0, sizeof(int16_t) * ch_im_in);
                    }
                    else
                    {
                        arm_u4_to_int16_reordered(
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 1),
                                                pBuffer,
                                                ch_im_in,
                                                z_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut = arm_nn_mat_mult_kernel_reordered_u4_int16_u2_icn(wt,
                                                bufferA,
                                                ch_im_out,
                                                ch_im_in * dim_kernel * dim_kernel,
                                                bias,
                                                pOut,
                                                z_wt,
                                                z_out,
                                                m_zero,
                                                n_zero);
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    /* check if there is left-over for compute */
    if (pBuffer != bufferA)
    {
        /* Negative N_ZERO Normalization */
        int8_t n_zero1;
        int8_t n_zero2;

        /* Offset over Weights */
        int16_t VzA[2] = {z_wt,z_wt};
        const int16_t *pzA = VzA;
        int32_t inzA = *__SIMD32(pzA);

        /* Weights Pointer */
        const uint8_t *pA = wt;
        int       i;

        int pOut_per_byte = 4;

        for (i = 0; i < ch_im_out; i++)
        {
            int32_t sum = bias[i];
            int16_t *pB = bufferA;

            uint16_t  colCnt = ch_im_in * dim_kernel * dim_kernel >> 3; // config.wt_data_t: u4 (8x uint4_t)

            /* accumulate over the vector */
            while (colCnt)
            {
                int32_t inA1, inA2;
                int32_t inA3, inA4;
                int32_t inB1, inB2;

                pA = (uint8_t *) read_and_pad_reordered_u4((void *)pA, &inA1, &inA2, &inA3, &inA4);

                inB1 = *__SIMD32(pB)++;
                inA1 = __SSUB16(inA1, inzA);
                inA2 = __SSUB16(inA2, inzA);
                sum = __SMLAD(inA1, inB1, sum);
                inB2 = *__SIMD32(pB)++;
                sum = __SMLAD(inA2, inB2, sum);
                inB1 = *__SIMD32(pB)++;
                inA3 = __SSUB16(inA3, inzA);
                inA4 = __SSUB16(inA4, inzA);
                sum = __SMLAD(inA3, inB1, sum);
                inB2 = *__SIMD32(pB)++;
                sum = __SMLAD(inA4, inB2, sum);
                colCnt--;
            }

            colCnt = ch_im_in * dim_kernel * dim_kernel & 0x7;; // config.wt_data_t: u4 (8x uint4_t)

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
                inA1 -= z_wt;
                sum += inA1 * inB1;
                colCnt--;
            }

            /* Normalize by ICN (u2 output) */
            __n_zero_negative_normalization(n_zero[i],&n_zero1,&n_zero2);
            sum  = ((__HI_SMULL(sum << n_zero1 ,m_zero[i])) >> n_zero2) + z_out;

            /* Store Outputs (u2 output) */
            switch(pOut_per_byte){
                case 4:
                    *pOut  = ( __USAT(sum, 2) );
                    pOut_per_byte--;
                    break;
                case 3:
                    *pOut |= ( __USAT(sum, 2) << 2);
                    pOut_per_byte--;
                    break;
                case 2:
                    *pOut |= ( __USAT(sum, 2) << 4);
                    pOut_per_byte--;
                    break;
                case 1:
                    *pOut++ |= ( __USAT(sum, 2) << 6);
                    pOut_per_byte-=4;
                    break;
            }
        }
    }

#else
#error "Cortex-M0 and Cortex-M3 not supported"
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
#endif /* ARM_MATH_DSP */

    /* Return to application */
    return ARM_MATH_SUCCESS;
}

/**
 * @} end of CMIXConv group
 */
