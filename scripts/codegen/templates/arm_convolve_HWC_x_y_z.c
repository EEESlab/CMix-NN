${config.header_top}
 * Description:  Mixed Precision Convolutional function that uses ${config.in_data_t}
 *               activations, ${config.wt_data_t} weights and produce ${config.out_data_t}
 *               output activations. Outputs are quantized using ${config.folding}
 *               folding technique.
${config.header_bottom}

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
   * @brief Mixed Precision Convolution ${config.folding} (in: ${config.in_data_t}, out: ${config.out_data_t}, wt: ${config.wt_data_t})
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
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
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
${config.fn_name}(const uint8_t *Im_in,
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
% if config.quantization == "PACT":
                    const uint8_t z_wt,
% elif config.quantization == "PACT_CH":
                    const uint8_t *z_wt,
% endif
% if config.folding == "thr":
                    const int16_t *thresholds,
% else:
                    const uint8_t z_out,
%   if config.folding == "icn":
                    const int32_t *m_zero,
                    const int8_t *n_zero,
%   else:
                    const int32_t m_zero,
                    const int8_t n_zero,
%   endif
% endif
                    int16_t * bufferA,
                    uint8_t *bufferB)
{

#if defined(ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t i_out_y, i_out_x, i_ker_y, i_ker_x;
    int16_t *pBuffer = bufferA;
    uint8_t *pOut = Im_out;

    if (ch_im_in % ${config.ch_in_constrain} != 0 || ch_im_out % ${config.ch_out_constrain} != 0)
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
                        ${config.reordered_no_shift_load_fn}(
% if config.in_data_t == 'u8':
                                                Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in,
% elif config.in_data_t == 'u4':
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 1),
% elif config.in_data_t == 'u2':
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 2),
% endif
                                                pBuffer,
                                                ch_im_in,
                                                z_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut = ${config.nn_mat_mul_fn}(wt,
                                                bufferA,
                                                ch_im_out,
                                                ch_im_in * dim_kernel * dim_kernel,
                                                bias,
                                                pOut,
                                                z_wt,
% if config.folding == "thr":
                                                thresholds);
% else:
                                                z_out,
                                                m_zero,
                                                n_zero);
% endif
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
                        ${config.reordered_no_shift_load_fn}(
% if config.in_data_t == 'u8':
                                                Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in,
% elif config.in_data_t == 'u4':
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 1),
% elif config.in_data_t == 'u2':
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 2),
% endif
                                                pBuffer,
                                                ch_im_in,
                                                z_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut = ${config.nn_mat_mul_fn}(wt,
                                                bufferA,
                                                ch_im_out,
                                                ch_im_in * dim_kernel * dim_kernel,
                                                bias,
                                                pOut,
                                                z_wt,
% if config.folding == "thr":
                                                thresholds);
% else:
                                                z_out,
                                                m_zero,
                                                n_zero);
% endif
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
                ${config.reordered_no_shift_load_fn}(
% if config.in_data_t == 'u8':
                                                Im_in + (i_ker_y * dim_im_in + i_out_x * stride - top_padding) * ch_im_in,
% elif config.in_data_t == 'u4':
                                                Im_in + (((i_ker_y * dim_im_in + i_out_x * stride - top_padding) * ch_im_in) >> 1),
% elif config.in_data_t == 'u2':
                                                Im_in + (((i_ker_y * dim_im_in + i_out_x * stride - top_padding) * ch_im_in) >> 2),
% endif
                                                pBuffer,
                                                ch_im_in * dim_kernel,
                                                z_in);
                pBuffer += ch_im_in * dim_kernel;
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut = ${config.nn_mat_mul_fn}(wt,
                                                bufferA,
                                                ch_im_out,
                                                ch_im_in * dim_kernel * dim_kernel,
                                                bias,
                                                pOut,
                                                z_wt,
% if config.folding == "thr":
                                                thresholds);
% else:
                                                z_out,
                                                m_zero,
                                                n_zero);
% endif
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
                        ${config.reordered_no_shift_load_fn}(
% if config.in_data_t == 'u8':
                                                Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in,
% elif config.in_data_t == 'u4':
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 1),
% elif config.in_data_t == 'u2':
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 2),
% endif
                                                pBuffer,
                                                ch_im_in,
                                                z_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut = ${config.nn_mat_mul_fn}(wt,
                                                bufferA,
                                                ch_im_out,
                                                ch_im_in * dim_kernel * dim_kernel,
                                                bias,
                                                pOut,
                                                z_wt,
% if config.folding == "thr":
                                                thresholds);
% else:
                                                z_out,
                                                m_zero,
                                                n_zero);
% endif
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
                        ${config.reordered_no_shift_load_fn}(
% if config.in_data_t == 'u8':
                                                Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in,
% elif config.in_data_t == 'u4':
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 1),
% elif config.in_data_t == 'u2':
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 2),
% endif
                                                pBuffer,
                                                ch_im_in,
                                                z_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut = ${config.nn_mat_mul_fn}(wt,
                                                bufferA,
                                                ch_im_out,
                                                ch_im_in * dim_kernel * dim_kernel,
                                                bias,
                                                pOut,
                                                z_wt,
% if config.folding == "thr":
                                                thresholds);
% else:
                                                z_out,
                                                m_zero,
                                                n_zero);
% endif
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    /* check if there is left-over for compute */
    if (pBuffer != bufferA)
    {
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

% if config.quantization=="PACT":
        /* Offset over Weights */
        int16_t VzA[2] = {z_wt,z_wt};
        const int16_t *pzA = VzA;
        int32_t inzA = *__SIMD32(pzA);
% endif

        /* Weights Pointer */
        const uint8_t *pA = wt;
        int       i;

% if config.out_data_t=='u4':
        int pOut_per_byte = 2;
% elif config.out_data_t=='u2':
        int pOut_per_byte = 4;
% endif

        for (i = 0; i < ch_im_out; i++)
        {
% if config.quantization=="PACT_CH":
            /* Offset over Weights */
            int16_t VzA[2] = {z_wt[i], z_wt[i]};
            const int16_t *pzA = VzA;
            int32_t inzA = *__SIMD32(pzA);
% endif
            int32_t sum = bias[i];
            int16_t *pB = bufferA;

% if config.wt_data_t=='u8':
            uint16_t  colCnt = ch_im_in * dim_kernel * dim_kernel >> 2; // config.wt_data_t: u4 (4x uint8_t)
% elif config.wt_data_t=='u4':
            uint16_t  colCnt = ch_im_in * dim_kernel * dim_kernel >> 3; // config.wt_data_t: u4 (8x uint4_t)
% elif config.wt_data_t=='u2':
            uint16_t  colCnt = ch_im_in * dim_kernel * dim_kernel >> 4; // config.wt_data_t: u2 (16x uint2_t)
% endif

            /* accumulate over the vector */
            while (colCnt)
            {
                int32_t inA1, inA2;
% if config.wt_data_t=='u4':
                int32_t inA3, inA4;
% elif config.wt_data_t=='u2':
                int32_t inA3, inA4;
                int32_t inA5, inA6;
                int32_t inA7, inA8;
% endif                
                int32_t inB1, inB2;

% if config.wt_data_t=='u8':
                pA = (uint8_t *) read_and_pad_reordered_u8((void *)pA, &inA1, &inA2);
% elif config.wt_data_t=='u4':
                pA = (uint8_t *) read_and_pad_reordered_u4((void *)pA, &inA1, &inA2, &inA3, &inA4);
% elif config.wt_data_t=='u2':
                pA = (uint8_t *) read_and_pad_reordered_u2((void *)pA, &inA1, &inA2, &inA3, &inA4, &inA5, &inA6, &inA7, &inA8);
% endif

                inB1 = *__SIMD32(pB)++;
                inA1 = __SSUB16(inA1, inzA);
                inA2 = __SSUB16(inA2, inzA);
                sum = __SMLAD(inA1, inB1, sum);
                inB2 = *__SIMD32(pB)++;
                sum = __SMLAD(inA2, inB2, sum);
% if config.wt_data_t=='u4' or config.wt_data_t=='u2':
                inB1 = *__SIMD32(pB)++;
                inA3 = __SSUB16(inA3, inzA);
                inA4 = __SSUB16(inA4, inzA);
                sum = __SMLAD(inA3, inB1, sum);
                inB2 = *__SIMD32(pB)++;
                sum = __SMLAD(inA4, inB2, sum);
% endif
% if config.wt_data_t=='u2':
                inB1 = *__SIMD32(pB)++;
                inA5 = __SSUB16(inA5, inzA);
                inA6 = __SSUB16(inA6, inzA);
                sum = __SMLAD(inA5, inB1, sum);
                inB2 = *__SIMD32(pB)++;
                sum = __SMLAD(inA6, inB2, sum);
                inB1 = *__SIMD32(pB)++;
                inA7 = __SSUB16(inA7, inzA);
                inA8 = __SSUB16(inA8, inzA);
                sum = __SMLAD(inA7, inB1, sum);
                inB2 = *__SIMD32(pB)++;
                sum = __SMLAD(inA8, inB2, sum);
% endif
                colCnt--;
            }

% if config.wt_data_t=='u8':
            colCnt = ch_im_in * dim_kernel * dim_kernel & 0x3; // config.wt_data_t: u4 (4x uint8_t)
% elif config.wt_data_t=='u4':
            colCnt = ch_im_in * dim_kernel * dim_kernel & 0x7;; // config.wt_data_t: u4 (8x uint4_t)
% elif config.wt_data_t=='u2':
            colCnt = ch_im_in * dim_kernel * dim_kernel & 0xf; // config.wt_data_t: u2 (16x uint2_t)
% endif

% if config.wt_data_t=='u4':
            int wt_per_byte = 2;
% elif config.wt_data_t=='u2':
            int wt_per_byte = 4;
% endif
            while (colCnt)
            {
                uint8_t inB1 = (uint8_t) *pB++;
                uint8_t inA1;
% if config.wt_data_t=='u8':
                inA1 = (uint8_t)*pA++;
% elif config.wt_data_t=='u4':
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
% elif config.wt_data_t=='u2':
                switch(wt_per_byte)
                {
                    case 4:
                        inA1 = (uint8_t) __USAT(*pA, 2);
                        break;
                    case 3:
                        inA1 = (uint8_t) __USAT(__ROR(*pA, 2), 2);
                        break;
                    case 2:
                        inA1 = (uint8_t) __USAT(__ROR(*pA, 2), 4);
                        break;
                    case 1:
                        inA1 = (uint8_t) __USAT(__ROR(*pA, 2), 6);
                        pA++;
                        break;
                }
% endif
% if config.quantization=="PACT":
                inA1 -= z_wt;
% elif config.quantization=="PACT_CH":
                inA1 -= z_wt[i];
% endif
                sum += inA1 * inB1;
                colCnt--;
            }

% if config.folding=="thr":
            /* Normalize by Thresholds (${config.out_data_t} output) */
%   if config.out_data_t=='u8':
            #warning No threasholds available at u8
%   elif config.out_data_t=='u4':
            sum = __int16_to_u4((int16_t) sum, &thresholds[(i)<<4]);
%   elif config.out_data_t=='u2':
            sum = __int16_to_u2((int16_t) sum , &thresholds[(i)<<2]);
%   endif
% elif config.folding=="icn":
            /* Normalize by ICN (${config.out_data_t} output) */
            __n_zero_negative_normalization(n_zero[i],&n_zero1,&n_zero2);
            sum  = ((__HI_SMULL(sum << n_zero1 ,m_zero[i])) >> n_zero2) + z_out;
% elif config.folding=="weights":
            /* Normalize by PACT+FW (${config.out_data_t} output) */
            sum  = ((__HI_SMULL(sum << n_zero1,m_zero)) >> n_zero2) + z_out;
% endif

            /* Store Outputs (${config.out_data_t} output) */
% if config.out_data_t=='u8':
            *pOut++ = (uint8_t) __USAT(sum, 8);
% elif config.out_data_t=='u4':
            switch(pOut_per_byte){
                case 2:
                    *pOut  = ( __USAT(sum, 4) );
                    pOut_per_byte--;
                    break;
                case 1:
                    *pOut++ |= ( __USAT(sum, 4) << 4 );
                    pOut_per_byte=2;
                    break;
            }
% elif config.out_data_t=='u2':
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
% endif
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
