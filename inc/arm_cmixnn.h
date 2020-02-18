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
 * Description:  Public header file for CMix-NN Library.
 * 
 * Target:       ARM Cortex-M cores
 *
 * $Date:        10 February 2020
 * $Revision:    Release v1.0.0
 *
 * $Authors:     Alessandro Capotondi
 *                <alessandro.capotondi AT unibo.it>
 *               Marco Fariselli
 *                <marco.fariselli AT greenwaves-technologies.com>
 *               Manuele Rusci 
 *                <manuele.rusci AT unibo.it>
 *
 ******************************************************************/

/**
   \mainpage CMIX-NN Software Library
   * 
   * To Be Completed
   * 
   */

/**
 * @defgroup CMIXNN CMix-NN Neural Network Functions
 * 
 * To Be Completed
 * 
 */

#ifndef _ARM_CMIXNN_H
#define _ARM_CMIXNN_H

#include "arm_math.h"
#include "arm_cmixnn_support.h"

#define USE_INTRINSIC

#ifdef __cplusplus
extern    "C"
{
#endif

/**
 * @defgroup CMIXConv CMix-NN Convolution Functions
 * 
 * To be completed.
 * 
 */

  /**
   * @brief Mixed Precision Convolution weights (in: u8, out: u8, wt: u8)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u8_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u8, wt: u8)
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
arm_convolve_HWC_u8_u8_u8_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u8, wt: u8)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u8_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u8, out: u8, wt: u4)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u8_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u8, wt: u4)
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
arm_convolve_HWC_u8_u8_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u8, wt: u4)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u8_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u8, out: u8, wt: u2)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u8_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u8, wt: u2)
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
arm_convolve_HWC_u8_u8_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u8, wt: u2)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u8_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u8, out: u4, wt: u8)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u4_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u4, wt: u8)
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
arm_convolve_HWC_u8_u4_u8_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u4, wt: u8)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u4_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u8, out: u4, wt: u4)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u4_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u4, wt: u4)
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
arm_convolve_HWC_u8_u4_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u4, wt: u4)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u4_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u8, out: u4, wt: u2)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u4_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u4, wt: u2)
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
arm_convolve_HWC_u8_u4_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u4, wt: u2)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u4_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u8, out: u2, wt: u8)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u2_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u2, wt: u8)
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
arm_convolve_HWC_u8_u2_u8_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u2, wt: u8)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u2_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u8, out: u2, wt: u4)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u2_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u2, wt: u4)
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
arm_convolve_HWC_u8_u2_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u2, wt: u4)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u2_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u8, out: u2, wt: u2)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u2_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u2, wt: u2)
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
arm_convolve_HWC_u8_u2_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u8, out: u2, wt: u2)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u8_u2_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u4, out: u8, wt: u8)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u8_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u4, out: u8, wt: u8)
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
arm_convolve_HWC_u4_u8_u8_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u4, out: u8, wt: u8)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u8_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u4, out: u8, wt: u4)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u8_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u4, out: u8, wt: u4)
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
arm_convolve_HWC_u4_u8_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u4, out: u8, wt: u4)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u8_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u4, out: u8, wt: u2)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u8_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u4, out: u8, wt: u2)
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
arm_convolve_HWC_u4_u8_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u4, out: u8, wt: u2)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u8_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u4, out: u4, wt: u8)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u4_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u4, out: u4, wt: u8)
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
arm_convolve_HWC_u4_u4_u8_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u4, out: u4, wt: u8)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u4_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u4, out: u4, wt: u4)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u4_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u4, out: u4, wt: u4)
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
arm_convolve_HWC_u4_u4_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u4, out: u4, wt: u4)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u4_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u4, out: u4, wt: u2)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u4_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u4, out: u4, wt: u2)
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
arm_convolve_HWC_u4_u4_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u4, out: u4, wt: u2)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u4_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u4, out: u2, wt: u8)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u2_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u4, out: u2, wt: u8)
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
arm_convolve_HWC_u4_u2_u8_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u4, out: u2, wt: u8)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u2_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u4, out: u2, wt: u4)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u2_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

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
arm_convolve_HWC_u4_u2_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u2_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u4, out: u2, wt: u2)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u2_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u4, out: u2, wt: u2)
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
arm_convolve_HWC_u4_u2_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u4, out: u2, wt: u2)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u4_u2_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u2, out: u8, wt: u8)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u8_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u8, wt: u8)
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
arm_convolve_HWC_u2_u8_u8_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u8, wt: u8)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u8_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u2, out: u8, wt: u4)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u8_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u8, wt: u4)
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
arm_convolve_HWC_u2_u8_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u8, wt: u4)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u8_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u2, out: u8, wt: u2)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u8_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u8, wt: u2)
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
arm_convolve_HWC_u2_u8_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u8, wt: u2)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u8_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u2, out: u4, wt: u8)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u4_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u4, wt: u8)
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
arm_convolve_HWC_u2_u4_u8_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u4, wt: u8)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u4_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u2, out: u4, wt: u4)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u4_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u4, wt: u4)
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
arm_convolve_HWC_u2_u4_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u4, wt: u4)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u4_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u2, out: u4, wt: u2)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u4_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u4, wt: u2)
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
arm_convolve_HWC_u2_u4_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u4, wt: u2)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u4_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u2, out: u2, wt: u8)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u2_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u2, wt: u8)
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
arm_convolve_HWC_u2_u2_u8_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u2, wt: u8)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u2_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u2, out: u2, wt: u4)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u2_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u2, wt: u4)
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
arm_convolve_HWC_u2_u2_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u2, wt: u4)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u2_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution weights (in: u2, out: u2, wt: u2)
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u2_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u2, wt: u2)
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
arm_convolve_HWC_u2_u2_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

  /**
   * @brief Mixed Precision Convolution icn (in: u2, out: u2, wt: u2)
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_convolve_HWC_u2_u2_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u8 weights
   *        and produce u8 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u8_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

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
                        uint8_t * bufferB);

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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u8_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u4 weights
   *        and produce u8 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u8_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u4 weights
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
arm_depthwise_separable_conv_HWC_u8_u8_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u4 weights
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u8_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u2 weights
   *        and produce u8 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u8_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u2 weights
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
arm_depthwise_separable_conv_HWC_u8_u8_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u2 weights
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u8_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u8 weights
   *        and produce u8 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u4_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

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
arm_depthwise_separable_conv_HWC_u8_u4_u8_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u4_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u4 weights
   *        and produce u8 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u4_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u4 weights
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
arm_depthwise_separable_conv_HWC_u8_u4_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u4 weights
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u4_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u2 weights
   *        and produce u8 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u4_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u2 weights
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
arm_depthwise_separable_conv_HWC_u8_u4_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u2 weights
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u4_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u8 weights
   *        and produce u8 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u2_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

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
arm_depthwise_separable_conv_HWC_u8_u2_u8_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u2_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u4 weights
   *        and produce u8 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u2_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u4 weights
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
arm_depthwise_separable_conv_HWC_u8_u2_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u4 weights
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u2_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u2 weights
   *        and produce u8 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u2_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u2 weights
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
arm_depthwise_separable_conv_HWC_u8_u2_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u8 activations, u2 weights
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u8_u2_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u8 weights
   *        and produce u4 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u8_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u8 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u4_u8_u8_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u8 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u8_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u4 weights
   *        and produce u4 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u8_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u4 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u4_u8_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u4 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u8_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u2 weights
   *        and produce u4 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u8_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u2 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u4_u8_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u2 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u8_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u8 weights
   *        and produce u4 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u4_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u8 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u4_u4_u8_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u8 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u4_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u4 weights
   *        and produce u4 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u4_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u4 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u4_u4_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u4 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u4_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u2 weights
   *        and produce u4 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u4_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u2 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u4_u4_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u2 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u4_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u8 weights
   *        and produce u4 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u2_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u8 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u4_u2_u8_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u8 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u2_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u4 weights
   *        and produce u4 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u2_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u4 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u4_u2_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u4 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u2_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u2 weights
   *        and produce u4 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u2_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u2 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u4_u2_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u4 activations, u2 weights
   *        and produce u4 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u4_u2_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u8 weights
   *        and produce u2 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u8_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u8 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u2_u8_u8_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u8 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u8_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u4 weights
   *        and produce u2 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u8_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u4 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u2_u8_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u4 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u8_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u2 weights
   *        and produce u2 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u8_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u2 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u2_u8_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u2 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u8_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u8 weights
   *        and produce u2 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u4_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u8 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u2_u4_u8_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u8 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u4_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u4 weights
   *        and produce u2 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u4_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u4 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u2_u4_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u4 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u4_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u2 weights
   *        and produce u2 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u4_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u2 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u2_u4_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u2 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u4_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u8 weights
   *        and produce u2 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u2_u8(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u8 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u2_u2_u8_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u8 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u2_u8_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u4 weights
   *        and produce u2 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u2_u4(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u4 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u2_u2_u4_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u4 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u2_u4_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u2 weights
   *        and produce u2 output activations. Outputs are quantized using weights folding technique.
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
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u2_u2(const uint8_t * Im_in,
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
                        const int32_t m_zero,
                        const int8_t n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u2 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
arm_depthwise_separable_conv_HWC_u2_u2_u2_icn(const uint8_t * Im_in,
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
                        uint8_t * bufferB);

/**
   * @brief Mixed Precision Depthwise Convolutional function that uses u2 activations, u2 weights
   *        and produce u2 output activations. Outputs are quantized using icn folding technique.
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
   * @param[in]       *z_wt       weights offset, per-output channel
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */

arm_status
arm_depthwise_separable_conv_HWC_u2_u2_u2_PACT_CH_icn(const uint8_t * Im_in,
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
                        const uint8_t *z_wt,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
                        int16_t * bufferA,
                        uint8_t * bufferB);

 /**
   * @brief Matrix-Multiplication function for u8 x int16_t convolution with reordered columns.
   *        Output is then quantized to u8 using weights config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       z_a         A operand offset
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u8_int16_u8(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t m_zero,
                        const int8_t n_zero);

 /**
   * @brief Matrix-Multiplication function for u8 x int16_t convolution with reordered columns.
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

uint8_t *
arm_nn_mat_mult_kernel_reordered_u8_int16_u8_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);

 /**
   * @brief Matrix-Multiplication function for u8 x int16_t convolution with reordered columns.
   *        Output is then quantized to u8 using icn config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       *z_a        pointer to A operand offsets (per-output channel)
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u8_int16_u8_PACT_CH_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t *z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);

 /**
   * @brief Matrix-Multiplication function for u4 x int16_t convolution with reordered columns.
   *        Output is then quantized to u8 using weights config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       z_a         A operand offset
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u4_int16_u8(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t m_zero,
                        const int8_t n_zero);

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

uint8_t *
arm_nn_mat_mult_kernel_reordered_u4_int16_u8_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);

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
   * @param[in]       *z_a        pointer to A operand offsets (per-output channel)
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u4_int16_u8_PACT_CH_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t *z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);

 /**
   * @brief Matrix-Multiplication function for u2 x int16_t convolution with reordered columns.
   *        Output is then quantized to u8 using weights config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       z_a         A operand offset
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u2_int16_u8(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t m_zero,
                        const int8_t n_zero);

 /**
   * @brief Matrix-Multiplication function for u2 x int16_t convolution with reordered columns.
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

uint8_t *
arm_nn_mat_mult_kernel_reordered_u2_int16_u8_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);

 /**
   * @brief Matrix-Multiplication function for u2 x int16_t convolution with reordered columns.
   *        Output is then quantized to u8 using icn config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       *z_a        pointer to A operand offsets (per-output channel)
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u2_int16_u8_PACT_CH_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t *z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);

 /**
   * @brief Matrix-Multiplication function for u8 x int16_t convolution with reordered columns.
   *        Output is then quantized to u4 using weights config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       z_a         A operand offset
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u8_int16_u4(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t m_zero,
                        const int8_t n_zero);

 /**
   * @brief Matrix-Multiplication function for u8 x int16_t convolution with reordered columns.
   *        Output is then quantized to u4 using icn config.folding technique.
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

uint8_t *
arm_nn_mat_mult_kernel_reordered_u8_int16_u4_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);

 /**
   * @brief Matrix-Multiplication function for u8 x int16_t convolution with reordered columns.
   *        Output is then quantized to u4 using icn config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       *z_a        pointer to A operand offsets (per-output channel)
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u8_int16_u4_PACT_CH_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t *z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);

 /**
   * @brief Matrix-Multiplication function for u4 x int16_t convolution with reordered columns.
   *        Output is then quantized to u4 using weights config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       z_a         A operand offset
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u4_int16_u4(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t m_zero,
                        const int8_t n_zero);

 /**
   * @brief Matrix-Multiplication function for u4 x int16_t convolution with reordered columns.
   *        Output is then quantized to u4 using icn config.folding technique.
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

uint8_t *
arm_nn_mat_mult_kernel_reordered_u4_int16_u4_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);

 /**
   * @brief Matrix-Multiplication function for u4 x int16_t convolution with reordered columns.
   *        Output is then quantized to u4 using icn config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       *z_a        pointer to A operand offsets (per-output channel)
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u4_int16_u4_PACT_CH_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t *z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);

 /**
   * @brief Matrix-Multiplication function for u2 x int16_t convolution with reordered columns.
   *        Output is then quantized to u4 using weights config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       z_a         A operand offset
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u2_int16_u4(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t m_zero,
                        const int8_t n_zero);

 /**
   * @brief Matrix-Multiplication function for u2 x int16_t convolution with reordered columns.
   *        Output is then quantized to u4 using icn config.folding technique.
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

uint8_t *
arm_nn_mat_mult_kernel_reordered_u2_int16_u4_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);

 /**
   * @brief Matrix-Multiplication function for u2 x int16_t convolution with reordered columns.
   *        Output is then quantized to u4 using icn config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       *z_a        pointer to A operand offsets (per-output channel)
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u2_int16_u4_PACT_CH_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t *z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);

 /**
   * @brief Matrix-Multiplication function for u8 x int16_t convolution with reordered columns.
   *        Output is then quantized to u2 using weights config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       z_a         A operand offset
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u8_int16_u2(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t m_zero,
                        const int8_t n_zero);

 /**
   * @brief Matrix-Multiplication function for u8 x int16_t convolution with reordered columns.
   *        Output is then quantized to u2 using icn config.folding technique.
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

uint8_t *
arm_nn_mat_mult_kernel_reordered_u8_int16_u2_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);

 /**
   * @brief Matrix-Multiplication function for u8 x int16_t convolution with reordered columns.
   *        Output is then quantized to u2 using icn config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       *z_a        pointer to A operand offsets (per-output channel)
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u8_int16_u2_PACT_CH_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t *z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);

 /**
   * @brief Matrix-Multiplication function for u4 x int16_t convolution with reordered columns.
   *        Output is then quantized to u2 using weights config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       z_a         A operand offset
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u4_int16_u2(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t m_zero,
                        const int8_t n_zero);

 /**
   * @brief Matrix-Multiplication function for u4 x int16_t convolution with reordered columns.
   *        Output is then quantized to u2 using icn config.folding technique.
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

uint8_t *
arm_nn_mat_mult_kernel_reordered_u4_int16_u2_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);

 /**
   * @brief Matrix-Multiplication function for u4 x int16_t convolution with reordered columns.
   *        Output is then quantized to u2 using icn config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       *z_a        pointer to A operand offsets (per-output channel)
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u4_int16_u2_PACT_CH_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t *z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);

 /**
   * @brief Matrix-Multiplication function for u2 x int16_t convolution with reordered columns.
   *        Output is then quantized to u2 using weights config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       z_a         A operand offset
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u2_int16_u2(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t m_zero,
                        const int8_t n_zero);

 /**
   * @brief Matrix-Multiplication function for u2 x int16_t convolution with reordered columns.
   *        Output is then quantized to u2 using icn config.folding technique.
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

uint8_t *
arm_nn_mat_mult_kernel_reordered_u2_int16_u2_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);

 /**
   * @brief Matrix-Multiplication function for u2 x int16_t convolution with reordered columns.
   *        Output is then quantized to u2 using icn config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
   * @param[in]       *z_a        pointer to A operand offsets (per-output channel)
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
arm_nn_mat_mult_kernel_reordered_u2_int16_u2_PACT_CH_icn(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
                        const uint8_t *z_a,
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);



#ifdef __cplusplus
}
#endif

/*
 * Other functions
 * To be completed.
 */

#ifdef __cplusplus
extern    "C"
{
#endif

/**
 * @defgroup CMIXActi CMix-NN Activation Functions
 *
 * To be completed.
 * 
 */

/**
 * @defgroup CMIXPool CMix-NN Pooling Functions
 *
 * To be completed.
 * 
 */

/**
 * @defgroup CMIXSoftmax CMix-NN Softmax Functions
 *
 * To be completed.
 * 
 */
#ifdef __cplusplus
}
#endif

#endif
