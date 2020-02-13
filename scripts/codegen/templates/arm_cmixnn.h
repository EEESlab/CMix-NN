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
${CMixNNAPI}

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
