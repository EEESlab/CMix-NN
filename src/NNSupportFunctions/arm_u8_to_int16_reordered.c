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
 * Title:        arm_u8_to_int16_reordered.c
 * Description:  Converts the elements of u8 vector to
 *               a reordered int16 vector (without left-shift).
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
 * @ingroup groupSupport
 */

/**
 * @addtogroup CMixNN_convert
 * @{
 */

/**
 * @brief Converts the elements of u8 vector to
 *        a reordered int16 vector (without left-shift).
 * @param[in] *pSrc       points to the u8 input vector
 * @param[out] *pDst      points to the int16 output vector
 * @param[in] blockSize   length of the input vector
 * @param[in] offset      input quantization offset
 * @return none.
 */

void
arm_u8_to_int16_reordered(
		const uint8_t *pSrc,
		int16_t *pDst,
		uint32_t blockSize,
	    const uint8_t offset)
{
    const uint8_t *pIn = pSrc;  /* Src pointer */
    uint32_t  blkCnt;           /* loop counter */
    int16_t offsets[2] = {offset, offset};
    const int16_t *offset_ptr = offsets;

#ifndef ARM_MATH_CM0_FAMILY
    int32_t   in;
    int32_t   in1, in2;
    int32_t   offset_vect = *__SIMD32(offset_ptr);

    /*loop Unrolling */
    blkCnt = blockSize >> 2u; // 4-elements block

    /* First part of the processing with loop unrolling.
       Second loop below computes the leftover */
    if(offset)
    {
		while (blkCnt > 0u)
		{
			in = *__SIMD32(pIn)++;

			in1 = __UXTB16(__ROR(in, 8));
			in2 = __UXTB16(in);

            in1 = __SSUB16(in1, offset_vect);
	        in2 = __SSUB16(in2, offset_vect);

#ifndef ARM_MATH_BIG_ENDIAN
            *__SIMD32(pDst)++ = in2;
            *__SIMD32(pDst)++ = in1;
#else
            *__SIMD32(pDst)++ = in1;
            *__SIMD32(pDst)++ = in2;
#endif
			/* Decrement the loop counter */
			blkCnt--;
		}
    }
    else
    {
		while (blkCnt > 0u)
		{
			in = *__SIMD32(pIn)++;

			in1 = __UXTB16(__ROR(in, 8));
			in2 = __UXTB16(in);

#ifndef ARM_MATH_BIG_ENDIAN
            *__SIMD32(pDst)++ = in2;
            *__SIMD32(pDst)++ = in1;
#else
            *__SIMD32(pDst)++ = in1;
            *__SIMD32(pDst)++ = in2;
#endif

			/* Decrement the loop counter */
			blkCnt--;
		}
    }

    /* If the blockSize is not a multiple, compute any remaining output samples here.
     ** No loop unrolling is used. */
    blkCnt = blockSize % 0x4u;
#else
#error "Cortex-M0 is not supported"
#endif /* #ifndef ARM_MATH_CM0_FAMILY */


    while (blkCnt > 0u)
    {
        *pDst++ = ((int16_t) * pIn++)-offset;

        /* Decrement the loop counter */
        blkCnt--;
    }
}

/**
 * @}
 */
