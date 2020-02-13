/*
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.
 * Modifications Copyright (C) 2018 University of Bologna
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
 * Project:      CMSIS NN Library - Mixed Precision INT-Q
 * Title:        ${config.filename}
 * Description:  Mixed Precision Depthwise Convolutional function
 *               that uses ${config.in_data_t} activations, ${config.wt_data_t} weights
 *               and produce ${config.in_data_t} output activations.
 *               Outputs are quantized using ${config.folding} folding technique.
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

#include "arm_math.h"
#include "arm_nnfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup NNConv
 * @{
 */

 /**
   * @brief Mixed Precision Depthwise Convolutional function that uses ${config.in_data_t} activations, ${config.wt_data_t} weights
   *        and produce ${config.in_data_t} output activations. Outputs are quantized using ${config.folding} folding technique.
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
% if config.quantization=="PACT":
   * @param[in]       z_wt        weights offset
% elif config.quantization=="PACT_CH":
   * @param[in]       *z_wt       weights offset, per-output channel
% endif
% if config.folding=="thr":
   * @param[in]       thresholds  pointer to thresholds
% elif config.folding=="icn":
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
% else:
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
% endif
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
${config.fn_name}(const uint8_t * Im_in,
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
% if config.quantization=="PACT":
                        const uint8_t z_wt,
% elif config.quantization=="PACT_CH":
                        const uint8_t *z_wt,
% endif
% if config.folding=="thr":
                        const int16_t * thresholds,
% else:
                        const uint8_t z_out,
%   if config.folding=="icn":
                        const int32_t *m_zero,
                        const int8_t *n_zero,
%   else:
                        const int32_t m_zero,
                        const int8_t n_zero,
%   endif
% endif
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

% if config.quantization=="PACT":
    int16_t Vz_wt[2] = {z_wt,z_wt};
    const int32_t *pz_wt = (int32_t *)Vz_wt;
    int32_t inz_wt = *__SIMD32(pz_wt);
% endif

    int16_t Vz_in[2] = {z_in,z_in};
    const int32_t *pz_in = (int32_t *) Vz_in;
    int32_t inz_in = *__SIMD32(pz_in);

% if config.folding == "weights":
    /* Negative N_ZERO Normalization */
    int8_t n_zero1;
    int8_t n_zero2;
    __n_zero_negative_normalization(n_zero,&n_zero1,&n_zero2);
% elif config.folding == "icn":
    /* Negative N_ZERO Normalization */
    int8_t n_zero1;
    int8_t n_zero2;
% endif

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
% if config.in_data_t=='u8':
                        memcpy(pBuffer, Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, ch_im_in);
% elif config.in_data_t=='u4':
                        /* Unpack Input uint4_t to uint8_t */
                        const uint8_t *Im_in2 =  Im_in + (i_ker_y * (dim_im_in) + i_ker_x) * (ch_im_in>>1);
                        for(int j=0, jj=0; j<ch_im_in>>1; j++)
                        {
                            pBuffer[jj++] =  Im_in2[j]       & 0x0F;
                            pBuffer[jj++] = (Im_in2[j] >> 4) & 0x0F;
                        }
% elif config.in_data_t=='u2':
                        /* Unpack Input uint4_t to uint8_t */
                        const uint8_t *Im_in2 =  Im_in + (i_ker_y * (dim_im_in) + i_ker_x) * (ch_im_in>>2);
                        for(int j=0, jj=0; j<ch_im_in>>2; j++)
                        {
                            pBuffer[jj++] =  Im_in2[j]       & 0x03;
                            pBuffer[jj++] = (Im_in2[j] >> 2) & 0x03;
                            pBuffer[jj++] = (Im_in2[j] >> 4) & 0x03;
                            pBuffer[jj++] = (Im_in2[j] >> 6) & 0x03;
                        }
% endif
                    }
                    pBuffer += ch_im_in;
                }
            }

            /* we will do the computation here for each channel */
            rowCnt = ch_im_out >> 2;
            row_shift = 0;
            pBias = bias;
% if config.quantization=="PACT_CH" or config.folding!="weights":
            int32_t ch_out_id=0;
% endif

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
% if config.quantization=="PACT_CH":
                    int16_t Vz_wt[2] = {z_wt[ch_out_id],z_wt[ch_out_id]};
                    const int32_t *pz_wt = (int32_t *)Vz_wt;
                    int32_t inz_wt = *__SIMD32(pz_wt);
% endif
                    int32_t     inA1, inA2, inB1, inB2, opA, opB;
% if config.wt_data_t!='u8':
                    int32_t     inA3, inA4;
% endif

                    inB1 = *__SIMD32(pB);
                    pB += ch_im_in;
                    opB = *__SIMD32(pB);
                    pB += ch_im_in;
                    inB2 = __PKHTB(opB, inB1, 16);
                    inB1 = __PKHBT(inB1, opB, 16);

% if config.wt_data_t=='u8':
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

%   if config.quantization=="PACT_CH":
                    Vz_wt[0] = Vz_wt[1] = z_wt[ch_out_id+1];
                    inz_wt = *__SIMD32(pz_wt);
%   endif
                    opA = __SSUB16(opA, inz_wt);
                    opB = __SSUB16(opB, inz_in);
                    sum2 = __SMLAD(opA, opB, sum2);

                    //sum3
                    opA = __UXTB16(inA2);
                    opB = __UXTB16(inB2);

%   if config.quantization=="PACT_CH":
                    Vz_wt[0] = Vz_wt[1] = z_wt[ch_out_id+2];
                    inz_wt = *__SIMD32(pz_wt);
%   endif
                    opA = __SSUB16(opA, inz_wt);
                    opB = __SSUB16(opB, inz_in);
                    sum3 = __SMLAD(opA, opB, sum3);

                    //sum4
                    opA = __UXTB16(__ROR(inA2, 8));
                    opB = __UXTB16(__ROR(inB2, 8));

%   if config.quantization=="PACT_CH":
                    Vz_wt[0] = Vz_wt[1] = z_wt[ch_out_id+3];
                    inz_wt = *__SIMD32(pz_wt);
%   endif
                    opA = __SSUB16(opA, inz_wt);
                    opB = __SSUB16(opB, inz_in);
                    sum4 = __SMLAD(opA, opB, sum4);
% elif config.wt_data_t=='u4':
                    uint32_t tmp_inA1  = (uint32_t) *((uint16_t *)pA); //ch 0, ch1 | ch2, ch3
                    pA+=ch_im_in>>1;
                    uint32_t tmp_inA2 = (uint32_t) *((uint16_t *)pA); //ch 0', ch1' | ch2', ch3'
                    pA+=ch_im_in>>1;

                    inA1 = __USAT(tmp_inA1, 4) | (__USAT(tmp_inA2, 4) << 16);
                    inA2 = __USAT(__ROR(tmp_inA1, 4), 4) | (__USAT(__ROR(tmp_inA2, 4), 4) << 16);
                    inA3 = __USAT(__ROR(tmp_inA1, 8), 4) | (__USAT(__ROR(tmp_inA2, 8), 4) << 16);
                    inA4 = __USAT(__ROR(tmp_inA1, 12), 4) | (__USAT(__ROR(tmp_inA2, 12), 4) << 16);

                    //sum
                    opA = __UXTB16(inA1);
                    opB = __UXTB16(inB1);

                    opA = __SSUB16(opA, inz_wt);
                    opB = __SSUB16(opB, inz_in);
                    sum = __SMLAD(opA, opB, sum);

                    //sum2
                    opA = __UXTB16(inA2);
                    opB = __UXTB16(__ROR(inB1, 8));

%   if config.quantization=="PACT_CH":
                    Vz_wt[0] = Vz_wt[1] = z_wt[ch_out_id+1];
                    inz_wt = *__SIMD32(pz_wt);
%   endif
                    opA = __SSUB16(opA, inz_wt);
                    opB = __SSUB16(opB, inz_in);
                    sum2 = __SMLAD(opA, opB, sum2);

                    //sum3
                    opA = __UXTB16(inA3);
                    opB = __UXTB16(inB2);

%   if config.quantization=="PACT_CH":
                    Vz_wt[0] = Vz_wt[1] = z_wt[ch_out_id+2];
                    inz_wt = *__SIMD32(pz_wt);
%   endif
                    opA = __SSUB16(opA, inz_wt);
                    opB = __SSUB16(opB, inz_in);
                    sum3 = __SMLAD(opA, opB, sum3);

                    //sum4
                    opA = __UXTB16(inA4);
                    opB = __UXTB16(__ROR(inB2, 8));

%   if config.quantization=="PACT_CH":
                    Vz_wt[0] = Vz_wt[1] = z_wt[ch_out_id+3];
                    inz_wt = *__SIMD32(pz_wt);
%   endif
                    opA = __SSUB16(opA, inz_wt);
                    opB = __SSUB16(opB, inz_in);
                    sum4 = __SMLAD(opA, opB, sum4);
% elif config.wt_data_t=='u2':
                    uint32_t tmp_inA1  = (uint32_t) *pA; //ch 0, ch1, ch2, ch3 | ...
                    pA+=ch_im_in>>2;
                    uint32_t tmp_inA2 = (uint32_t) *pA; //ch 0', ch1', ch2', ch3' | ...
                    pA+=ch_im_in>>2;

                    inA1 = __USAT(tmp_inA1, 2) | (__USAT(tmp_inA2, 2) << 16);
                    inA2 = __USAT(__ROR(tmp_inA1, 2), 2) | (__USAT(__ROR(tmp_inA2, 2), 2) << 16);
                    inA3 = __USAT(__ROR(tmp_inA1, 4), 2) | (__USAT(__ROR(tmp_inA2, 4), 2) << 16);
                    inA4 = __USAT(__ROR(tmp_inA1, 6), 2) | (__USAT(__ROR(tmp_inA2, 6), 2) << 16);

                    //sum
                    opA = __UXTB16(inA1);
                    opB = __UXTB16(inB1);

                    opA = __SSUB16(opA, inz_wt);
                    opB = __SSUB16(opB, inz_in);
                    sum = __SMLAD(opA, opB, sum);

                    //sum2
                    opA = __UXTB16(inA2);
                    opB = __UXTB16(__ROR(inB1, 8));

%   if config.quantization=="PACT_CH":
                    Vz_wt[0] = Vz_wt[1] = z_wt[ch_out_id+1];
                    inz_wt = *__SIMD32(pz_wt);
%   endif
                    opA = __SSUB16(opA, inz_wt);
                    opB = __SSUB16(opB, inz_in);
                    sum2 = __SMLAD(opA, opB, sum2);

                    //sum3
                    opA = __UXTB16(inA3);
                    opB = __UXTB16(inB2);

%   if config.quantization=="PACT_CH":
                    Vz_wt[0] = Vz_wt[1] = z_wt[ch_out_id+2];
                    inz_wt = *__SIMD32(pz_wt);
%   endif
                    opA = __SSUB16(opA, inz_wt);
                    opB = __SSUB16(opB, inz_in);
                    sum3 = __SMLAD(opA, opB, sum3);

                    //sum4
                    opA = __UXTB16(inA4);
                    opB = __UXTB16(__ROR(inB2, 8));

%   if config.quantization=="PACT_CH":
                    Vz_wt[0] = Vz_wt[1] = z_wt[ch_out_id+3];
                    inz_wt = *__SIMD32(pz_wt);
%   endif
                    opA = __SSUB16(opA, inz_wt);
                    opB = __SSUB16(opB, inz_in);
                    sum4 = __SMLAD(opA, opB, sum4);
% endif
                    colCnt--;
                }
#else
#error Missing Big Endian Implementation
#endif /* ARM_MATH_BIG_ENDIAN */

                colCnt = (dim_kernel * dim_kernel) & 0x1;
                while (colCnt)
                {
                    union arm_nnword inA, inB;
% if config.wt_data_t=='u8':
                    inA.word = *__SIMD32(pA);
                    pA += ch_im_in;
% elif config.wt_data_t=='u4':
                    uint32_t tmp_inA = (uint32_t) *((uint16_t *)pA); //ch 0', ch1' | ch2', ch3'
                    pA += ch_im_in>>1;
                    inA.bytes[0] = (uint8_t) __USAT(tmp_inA, 4);
                    inA.bytes[1] = (uint8_t) __USAT(__ROR(tmp_inA, 4), 4);
                    inA.bytes[2] = (uint8_t) __USAT(__ROR(tmp_inA, 8), 4);
                    inA.bytes[3] = (uint8_t) __USAT(__ROR(tmp_inA, 12), 4);
% elif config.wt_data_t=='u2':
                   uint32_t tmp_inA = (uint32_t) *pA; //ch 0', ch1' | ch2', ch3'
                    pA += ch_im_in>>2;
                    inA.bytes[0] = (uint8_t) __USAT(tmp_inA, 2);
                    inA.bytes[1] = (uint8_t) __USAT(__ROR(tmp_inA, 2), 2);
                    inA.bytes[2] = (uint8_t) __USAT(__ROR(tmp_inA, 4), 2);
                    inA.bytes[3] = (uint8_t) __USAT(__ROR(tmp_inA, 6), 2);
% endif
                    inB.word = *__SIMD32(pB);
                    pB += ch_im_in;
% if config.quantization=="PACT_CH":
                    sum  += (((uint8_t) inA.bytes[0])-z_wt[ch_out_id]) * (((uint8_t) inB.bytes[0])-z_in);
                    sum2 += (((uint8_t) inA.bytes[1])-z_wt[ch_out_id+1]) * (((uint8_t) inB.bytes[1])-z_in);
                    sum3 += (((uint8_t) inA.bytes[2])-z_wt[ch_out_id+2]) * (((uint8_t) inB.bytes[2])-z_in);
                    sum4 += (((uint8_t) inA.bytes[3])-z_wt[ch_out_id+3]) * (((uint8_t) inB.bytes[3])-z_in);
% else:
                    sum  += (((uint8_t) inA.bytes[0])-z_wt) * (((uint8_t) inB.bytes[0])-z_in);
                    sum2 += (((uint8_t) inA.bytes[1])-z_wt) * (((uint8_t) inB.bytes[1])-z_in);
                    sum3 += (((uint8_t) inA.bytes[2])-z_wt) * (((uint8_t) inB.bytes[2])-z_in);
                    sum4 += (((uint8_t) inA.bytes[3])-z_wt) * (((uint8_t) inB.bytes[3])-z_in);
% endif
                    colCnt--;
                }

% if config.folding=="thr":
                /* Normalize by Thresholds (${config.out_data_t} output) */
%   if config.out_data_t=='u8': 
                #warning No threasholds available at u8
%   elif config.out_data_t=='u4':
                sum = __int16_to_u4((int16_t) sum, &thresholds[(ch_out_id++)<<4]);
                sum2 = __int16_to_u4((int16_t) sum2, &thresholds[(ch_out_id++)<<4]);
                sum3 = __int16_to_u4((int16_t) sum3, &thresholds[(ch_out_id++)<<4]);
                sum4 = __int16_to_u4((int16_t) sum4, &thresholds[(ch_out_id++)<<4]);
%   elif config.out_data_t=='u2':
                sum = __int16_to_u2((int16_t) sum , &thresholds[(ch_out_id++)<<2]);
                sum2 = __int16_to_u2((int16_t) sum2, &thresholds[(ch_out_id++)<<2]);
                sum3 = __int16_to_u2((int16_t) sum3 , &thresholds[(ch_out_id++)<<2]);
                sum4 = __int16_to_u2((int16_t) sum4 , &thresholds[(ch_out_id++)<<2]);
%   endif
% elif config.folding=="icn":
                /* Normalize by ICN (${config.out_data_t} output) */
                __n_zero_negative_normalization(n_zero[ch_out_id],&n_zero1,&n_zero2);
                sum  = ((__HI_SMULL(sum << n_zero1 ,m_zero[ch_out_id++])) >> n_zero2) + z_out;
                __n_zero_negative_normalization(n_zero[ch_out_id],&n_zero1,&n_zero2);
                sum2  = ((__HI_SMULL(sum2 << n_zero1 ,m_zero[ch_out_id++])) >> n_zero2) + z_out;
                __n_zero_negative_normalization(n_zero[ch_out_id],&n_zero1,&n_zero2);
                sum3  = ((__HI_SMULL(sum3 << n_zero1 ,m_zero[ch_out_id++])) >> n_zero2) + z_out;
                __n_zero_negative_normalization(n_zero[ch_out_id],&n_zero1,&n_zero2);
                sum4  = ((__HI_SMULL(sum4 << n_zero1 ,m_zero[ch_out_id++])) >> n_zero2) + z_out;
% elif config.folding=="weights":
                /* Normalize by PACT+FW (${config.out_data_t} output) */
                sum  = ((__HI_SMULL(sum << n_zero1,m_zero)) >> n_zero2) + z_out;
                sum2 = ((__HI_SMULL(sum2 << n_zero1,m_zero)) >> n_zero2) + z_out;
                sum3 = ((__HI_SMULL(sum3 << n_zero1,m_zero)) >> n_zero2) + z_out;
                sum4 = ((__HI_SMULL(sum4 << n_zero1,m_zero)) >> n_zero2) + z_out;
% endif

                /* Store Outputs (${config.out_data_t} output) */
% if config.out_data_t=='u8':
                *pOut++ = (uint8_t) __USAT(sum, 8);
                *pOut++ = (uint8_t) __USAT(sum2, 8);
                *pOut++ = (uint8_t) __USAT(sum3, 8);
                *pOut++ = (uint8_t) __USAT(sum4, 8);
% elif config.out_data_t=='u4':
                *pOut++ = ( __USAT(sum, 4) ) | ( __USAT(sum2, 4) << 4 );
                *pOut++ = ( __USAT(sum3, 4)) | ( __USAT(sum4, 4) << 4 );
% elif config.out_data_t=='u2':
                *pOut++ = ( __USAT(sum,2)) | ( __USAT(sum2,2) << 2 ) | ( __USAT(sum3,2) << 4 ) | ( __USAT(sum4,2) << 6 );
% endif

% if config.wt_data_t=='u8':
                row_shift += 4;
% elif config.wt_data_t=='u4':
                row_shift += 2;
% elif config.wt_data_t=='u2':
                row_shift += 1;
% endif
                rowCnt--;
            }

% if config.wt_data_t=='u4':
            int row_per_byte = 2;
            int row_shift_wt = row_shift;
% elif config.wt_data_t=='u2':
            int row_per_byte = 4;
            int row_shift_wt = row_shift;
% endif

% if config.out_data_t=='u4':
            int row_per_byte_out = 2;
% elif config.out_data_t=='u2':
            int row_per_byte_out = 4;
% endif
            rowCnt = ch_im_out & 0x3;
            while (rowCnt)
            {
                uint8_t     *pB = colBuffer + row_shift;
                int32_t     sum = *pBias++;
                uint16_t  colCnt = (dim_kernel * dim_kernel);

% if config.wt_data_t=='u8':
                const uint8_t *pA = wt + row_shift;
                row_shift += 1;
% elif config.wt_data_t=='u4':
                const uint8_t *pA = wt + row_shift_wt;
% elif config.wt_data_t=='u2':
                const uint8_t *pA = wt + row_shift_wt;
% endif

                while (colCnt)
                {
                    uint8_t B1 = *pB;
% if config.wt_data_t=='u8':
                    uint8_t A1 = (uint8_t) __USAT(*pA, 8);
                    pA += ch_im_in;
% elif config.wt_data_t=='u4':
                    uint8_t A1;
                    switch(row_per_byte)
                    {
                        case 2:
                            A1 = (uint8_t) __USAT(*pA, 4);
                            break;
                        case 1:
                            A1 = (uint8_t) __USAT(__ROR(*pA, 4), 4);
                            break;
                    }
                    pA += ch_im_in>>1;
% elif config.wt_data_t=='u2':
                    uint8_t A1;
                    switch(row_per_byte)
                    {
                        case 4:
                            A1 = (uint8_t) __USAT(*pA, 2);
                            break;
                        case 3:
                            A1 = (uint8_t) __USAT(__ROR(*pA, 2), 2);
                            break;
                        case 2:
                            A1 = (uint8_t) __USAT(__ROR(*pA, 2), 4);
                            break;
                        case 1:
                            A1 = (uint8_t) __USAT(__ROR(*pA, 2), 6);
                            break;
                    }
                    pA += ch_im_in>>1;
% endif
                    pB += ch_im_in;

% if config.quantization=="PACT":
                    A1 -= z_wt;
% else:
                    A1 -= z_wt[ch_out_id];
% endif
                    B1 -= z_in;
                    sum += A1 * B1;

                    colCnt--;
                }

% if config.folding=="thr":
                /* Normalize by Thresholds (${config.out_data_t} output) */
%   if config.out_data_t=='u8': 
                #warning No threasholds available at u8
%   elif config.out_data_t=='u4':
                sum = __int16_to_u4((int16_t) sum, &thresholds[(ch_out_id++)<<4]);
%   elif config.out_data_t=='u2':
                sum = __int16_to_u2((int16_t) sum , &thresholds[(ch_out_id++)<<2]);
%   endif
% elif config.folding=="icn":
                /* Normalize by ICN (${config.out_data_t} output) */
                __n_zero_negative_normalization(n_zero[ch_out_id],&n_zero1,&n_zero2);
                sum  = ((__HI_SMULL(sum << n_zero1 ,m_zero[ch_out_id++])) >> n_zero2) + z_out;
% elif config.folding=="weights":
                /* Normalize by PACT+FW (${config.out_data_t} output) */
                sum  = ((__HI_SMULL(sum << n_zero1,m_zero)) >> n_zero2) + z_out;
% endif

                /* Store Outputs (${config.out_data_t} output) */
% if config.out_data_t=='u8':
                *pOut++ = (uint8_t) __USAT(sum, 8);
% elif config.out_data_t=='u4':
                switch(row_per_byte_out){
                    case 2:
                        *pOut  = ( __USAT(sum, 4) );
                        row_per_byte_out--;
                        break;
                    case 1:
                        *pOut++ |= ( __USAT(sum, 4) << 4 );
                        row_per_byte_out=2;
                        break;
                }
% elif config.out_data_t=='u2':
                switch(row_per_byte_out){
                    case 4:
                        *pOut  = ( __USAT(sum, 2) );
                        row_per_byte_out--;
                        break;
                    case 3:
                        *pOut |= ( __USAT(sum, 2) << 2);
                        row_per_byte_out--;
                        break;
                    case 2:
                        *pOut |= ( __USAT(sum, 2) << 4);
                        row_per_byte_out--;
                        break;
                    case 1:
                        *pOut++ |= ( __USAT(sum, 2) << 6);
                        row_per_byte_out-=4;
                        break;
                }
% endif
                rowCnt--;
% if config.wt_data_t=='u4':
                row_per_byte--;
                if(!row_per_byte){
                    row_per_byte=2;
                    row_shift_wt++;
                }
% elif config.wt_data_t=='u2':
                row_per_byte--;
                if(!row_per_byte){
                    row_per_byte=4;
                    row_shift_wt++;
                }
% endif
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
 * @} end of NNConv group
 */
