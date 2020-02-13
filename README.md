# CMix-NN: Mixed Low-Precision CNN Library for Memory-Constrained Edge Devices
*Alessandro Capotondi\*, Manuele Rusci\*\*, Marco Fariselli\*\*, Luca Benini\*\**

\*: Università di Modena e Reggio-Emilia, \*\*: Università di Bologna

## Keyworks
TinyML; Edge Computing; ARM Cortex-M; STM32H7; STM32F7; STM32L4; STM32F4

## Brief
Running inference tasks at the *edge* of the sensing infrastructure minimizes the user's network bandwidth and improves the response time. When envisioning *battery-powered IoT devices* with smart capabilities, the *limited energy* budget puts constraints on the design of computing platforms for edge computing. *Microcontroller Units (MCUs)*, which dominate the low-power spectrum of the computing platforms, typically feature *limited memory resources* (up to few MB of FLASH and below 1 MB of on-chip RAM) and *lack floating-point* hardware support. Hence, the deployment of high computational and memory requirements of deep learning workloads on edge devices results exceptionally challenging.

*Low-precision* integer arithmetic is a necessary ingredient for enabling Deep Learning inference on *tiny and resource-constrained IoT edge devices*. The approach described by [Rusci et al.](https://arxiv.org/abs/1905.13082) leveraged *heterogeneous mixed-precision* to deploy deep inference networks on *tiny MCUs*. The proposed technique aims at cutting the number of bits of individual weight or activation tensors below 8 bits up to fit the memory constraints and, at the same time, paying a limited accuracy drop if compared to the full-precision network. However, despite the numerous works addressing quantization on the server-side, no solution is provided for the deployment phase, especially if considering the mixed-precision sub-byte scenario. To tackle this problem, we present **CMix-NN**, an **open-source mixed-precision library** for quantized neural networks deployment on microcontroller targets. Differently, from state-of-the-art deployment solutions available for MCUs, our library supports convolutional kernels with *any bit precision** in the set of 8, 4 and 2 bits, for *any of the convolution operands*.

## Papers
Please, cite the following papers if you are going to use this library.
- **CMix-NN**
  *Accepted ISCAS20*

- **Mixed-Precision Training Methodology**
  *Rusci, Manuele, Alessandro Capotondi, and Luca Benini. "Memory-driven mixed low precision quantization for enabling deep network inference on microcontrollers." [arXiv preprint arXiv:1905.13082](https://arxiv.org/abs/1905.13082) (2019).*
  ```
  @misc{rusci2019memorydriven,
      title={Memory-Driven Mixed Low Precision Quantization For Enabling Deep Network Inference On Microcontrollers},
      author={Manuele Rusci and Alessandro Capotondi and Luca Benini},
      year={2019},
      eprint={1905.13082},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
  }
  ```

## Content
The CMix-NN is a C inference library for ARM Cortex-M MCU:
- **Include**: contains the public header file of CMix-NN
- **Source**: contains the implementation of convolutional kernels supported by CMix-NN
- **scripts/codegen**: contains the code genetator for the CMix-NN sources.

## On-going activities and Expected Contributions
- [ ] Add MobilenetV1 example *(coming soon)*
- [ ] Add dense functions *(coming soon)*
- [ ] Add the scripts to formart the parameters 
- [ ] Add pooling functions

## Supported Layers
|	Function	|	In Datatype	|	Out Datatype	|	Weights Datatype	|	Quantization Flavor	|	Filename	|
|	---	|	---	|	---	|	---	|	---	|	---	|
|	arm_convolve_HWC_u2_u2_u2	|	2-bit	|	2-bit	|	2-bit	|	PL	|	arm_convolve_HWC_u2_u2_u2.c	|
|	arm_convolve_HWC_u2_u2_u2_icn	|	2-bit	|	2-bit	|	2-bit	|	PL+ICN	|	arm_convolve_HWC_u2_u2_u2_icn.c	|
|	arm_convolve_HWC_u2_u2_u2_PACT_CH_icn	|	2-bit	|	2-bit	|	2-bit	|	PC+ICN	|	arm_convolve_HWC_u2_u2_u2_PACT_CH_icn.c	|
|	arm_convolve_HWC_u2_u2_u4	|	2-bit	|	2-bit	|	4-bit	|	PL	|	arm_convolve_HWC_u2_u2_u4.c	|
|	arm_convolve_HWC_u2_u2_u4_icn	|	2-bit	|	2-bit	|	4-bit	|	PL+ICN	|	arm_convolve_HWC_u2_u2_u4_icn.c	|
|	arm_convolve_HWC_u2_u2_u4_PACT_CH_icn	|	2-bit	|	2-bit	|	4-bit	|	PC+ICN	|	arm_convolve_HWC_u2_u2_u4_PACT_CH_icn.c	|
|	arm_convolve_HWC_u2_u2_u8	|	2-bit	|	2-bit	|	8-bit	|	PL	|	arm_convolve_HWC_u2_u2_u8.c	|
|	arm_convolve_HWC_u2_u2_u8_icn	|	2-bit	|	2-bit	|	8-bit	|	PL+ICN	|	arm_convolve_HWC_u2_u2_u8_icn.c	|
|	arm_convolve_HWC_u2_u2_u8_PACT_CH_icn	|	2-bit	|	2-bit	|	8-bit	|	PC+ICN	|	arm_convolve_HWC_u2_u2_u8_PACT_CH_icn.c	|
|	arm_convolve_HWC_u2_u4_u2	|	2-bit	|	4-bit	|	2-bit	|	PL	|	arm_convolve_HWC_u2_u4_u2.c	|
|	arm_convolve_HWC_u2_u4_u2_icn	|	2-bit	|	4-bit	|	2-bit	|	PL+ICN	|	arm_convolve_HWC_u2_u4_u2_icn.c	|
|	arm_convolve_HWC_u2_u4_u2_PACT_CH_icn	|	2-bit	|	4-bit	|	2-bit	|	PC+ICN	|	arm_convolve_HWC_u2_u4_u2_PACT_CH_icn.c	|
|	arm_convolve_HWC_u2_u4_u4	|	2-bit	|	4-bit	|	4-bit	|	PL	|	arm_convolve_HWC_u2_u4_u4.c	|
|	arm_convolve_HWC_u2_u4_u4_icn	|	2-bit	|	4-bit	|	4-bit	|	PL+ICN	|	arm_convolve_HWC_u2_u4_u4_icn.c	|
|	arm_convolve_HWC_u2_u4_u4_PACT_CH_icn	|	2-bit	|	4-bit	|	4-bit	|	PC+ICN	|	arm_convolve_HWC_u2_u4_u4_PACT_CH_icn.c	|
|	arm_convolve_HWC_u2_u4_u8	|	2-bit	|	4-bit	|	8-bit	|	PL	|	arm_convolve_HWC_u2_u4_u8.c	|
|	arm_convolve_HWC_u2_u4_u8_icn	|	2-bit	|	4-bit	|	8-bit	|	PL+ICN	|	arm_convolve_HWC_u2_u4_u8_icn.c	|
|	arm_convolve_HWC_u2_u4_u8_PACT_CH_icn	|	2-bit	|	4-bit	|	8-bit	|	PC+ICN	|	arm_convolve_HWC_u2_u4_u8_PACT_CH_icn.c	|
|	arm_convolve_HWC_u2_u8_u2	|	2-bit	|	8-bit	|	2-bit	|	PL	|	arm_convolve_HWC_u2_u8_u2.c	|
|	arm_convolve_HWC_u2_u8_u2_icn	|	2-bit	|	8-bit	|	2-bit	|	PL+ICN	|	arm_convolve_HWC_u2_u8_u2_icn.c	|
|	arm_convolve_HWC_u2_u8_u2_PACT_CH_icn	|	2-bit	|	8-bit	|	2-bit	|	PC+ICN	|	arm_convolve_HWC_u2_u8_u2_PACT_CH_icn.c	|
|	arm_convolve_HWC_u2_u8_u4	|	2-bit	|	8-bit	|	4-bit	|	PL	|	arm_convolve_HWC_u2_u8_u4.c	|
|	arm_convolve_HWC_u2_u8_u4_icn	|	2-bit	|	8-bit	|	4-bit	|	PL+ICN	|	arm_convolve_HWC_u2_u8_u4_icn.c	|
|	arm_convolve_HWC_u2_u8_u4_PACT_CH_icn	|	2-bit	|	8-bit	|	4-bit	|	PC+ICN	|	arm_convolve_HWC_u2_u8_u4_PACT_CH_icn.c	|
|	arm_convolve_HWC_u2_u8_u8	|	2-bit	|	8-bit	|	8-bit	|	PL	|	arm_convolve_HWC_u2_u8_u8.c	|
|	arm_convolve_HWC_u2_u8_u8_icn	|	2-bit	|	8-bit	|	8-bit	|	PL+ICN	|	arm_convolve_HWC_u2_u8_u8_icn.c	|
|	arm_convolve_HWC_u2_u8_u8_PACT_CH_icn	|	2-bit	|	8-bit	|	8-bit	|	PC+ICN	|	arm_convolve_HWC_u2_u8_u8_PACT_CH_icn.c	|
|	arm_convolve_HWC_u4_u2_u2	|	4-bit	|	2-bit	|	2-bit	|	PL	|	arm_convolve_HWC_u4_u2_u2.c	|
|	arm_convolve_HWC_u4_u2_u2_icn	|	4-bit	|	2-bit	|	2-bit	|	PL+ICN	|	arm_convolve_HWC_u4_u2_u2_icn.c	|
|	arm_convolve_HWC_u4_u2_u2_PACT_CH_icn	|	4-bit	|	2-bit	|	2-bit	|	PC+ICN	|	arm_convolve_HWC_u4_u2_u2_PACT_CH_icn.c	|
|	arm_convolve_HWC_u4_u2_u4	|	4-bit	|	2-bit	|	4-bit	|	PL	|	arm_convolve_HWC_u4_u2_u4.c	|
|	arm_convolve_HWC_u4_u2_u4_icn	|	4-bit	|	2-bit	|	4-bit	|	PL+ICN	|	arm_convolve_HWC_u4_u2_u4_icn.c	|
|	arm_convolve_HWC_u4_u2_u4_PACT_CH_icn	|	4-bit	|	2-bit	|	4-bit	|	PC+ICN	|	arm_convolve_HWC_u4_u2_u4_PACT_CH_icn.c	|
|	arm_convolve_HWC_u4_u2_u8	|	4-bit	|	2-bit	|	8-bit	|	PL	|	arm_convolve_HWC_u4_u2_u8.c	|
|	arm_convolve_HWC_u4_u2_u8_icn	|	4-bit	|	2-bit	|	8-bit	|	PL+ICN	|	arm_convolve_HWC_u4_u2_u8_icn.c	|
|	arm_convolve_HWC_u4_u2_u8_PACT_CH_icn	|	4-bit	|	2-bit	|	8-bit	|	PC+ICN	|	arm_convolve_HWC_u4_u2_u8_PACT_CH_icn.c	|
|	arm_convolve_HWC_u4_u4_u2	|	4-bit	|	4-bit	|	2-bit	|	PL	|	arm_convolve_HWC_u4_u4_u2.c	|
|	arm_convolve_HWC_u4_u4_u2_icn	|	4-bit	|	4-bit	|	2-bit	|	PL+ICN	|	arm_convolve_HWC_u4_u4_u2_icn.c	|
|	arm_convolve_HWC_u4_u4_u2_PACT_CH_icn	|	4-bit	|	4-bit	|	2-bit	|	PC+ICN	|	arm_convolve_HWC_u4_u4_u2_PACT_CH_icn.c	|
|	arm_convolve_HWC_u4_u4_u4	|	4-bit	|	4-bit	|	4-bit	|	PL	|	arm_convolve_HWC_u4_u4_u4.c	|
|	arm_convolve_HWC_u4_u4_u4_icn	|	4-bit	|	4-bit	|	4-bit	|	PL+ICN	|	arm_convolve_HWC_u4_u4_u4_icn.c	|
|	arm_convolve_HWC_u4_u4_u4_PACT_CH_icn	|	4-bit	|	4-bit	|	4-bit	|	PC+ICN	|	arm_convolve_HWC_u4_u4_u4_PACT_CH_icn.c	|
|	arm_convolve_HWC_u4_u4_u8	|	4-bit	|	4-bit	|	8-bit	|	PL	|	arm_convolve_HWC_u4_u4_u8.c	|
|	arm_convolve_HWC_u4_u4_u8_icn	|	4-bit	|	4-bit	|	8-bit	|	PL+ICN	|	arm_convolve_HWC_u4_u4_u8_icn.c	|
|	arm_convolve_HWC_u4_u4_u8_PACT_CH_icn	|	4-bit	|	4-bit	|	8-bit	|	PC+ICN	|	arm_convolve_HWC_u4_u4_u8_PACT_CH_icn.c	|
|	arm_convolve_HWC_u4_u8_u2	|	4-bit	|	8-bit	|	2-bit	|	PL	|	arm_convolve_HWC_u4_u8_u2.c	|
|	arm_convolve_HWC_u4_u8_u2_icn	|	4-bit	|	8-bit	|	2-bit	|	PL+ICN	|	arm_convolve_HWC_u4_u8_u2_icn.c	|
|	arm_convolve_HWC_u4_u8_u2_PACT_CH_icn	|	4-bit	|	8-bit	|	2-bit	|	PC+ICN	|	arm_convolve_HWC_u4_u8_u2_PACT_CH_icn.c	|
|	arm_convolve_HWC_u4_u8_u4	|	4-bit	|	8-bit	|	4-bit	|	PL	|	arm_convolve_HWC_u4_u8_u4.c	|
|	arm_convolve_HWC_u4_u8_u4_icn	|	4-bit	|	8-bit	|	4-bit	|	PL+ICN	|	arm_convolve_HWC_u4_u8_u4_icn.c	|
|	arm_convolve_HWC_u4_u8_u4_PACT_CH_icn	|	4-bit	|	8-bit	|	4-bit	|	PC+ICN	|	arm_convolve_HWC_u4_u8_u4_PACT_CH_icn.c	|
|	arm_convolve_HWC_u4_u8_u8	|	4-bit	|	8-bit	|	8-bit	|	PL	|	arm_convolve_HWC_u4_u8_u8.c	|
|	arm_convolve_HWC_u4_u8_u8_icn	|	4-bit	|	8-bit	|	8-bit	|	PL+ICN	|	arm_convolve_HWC_u4_u8_u8_icn.c	|
|	arm_convolve_HWC_u4_u8_u8_PACT_CH_icn	|	4-bit	|	8-bit	|	8-bit	|	PC+ICN	|	arm_convolve_HWC_u4_u8_u8_PACT_CH_icn.c	|
|	arm_convolve_HWC_u8_u2_u2	|	8-bit	|	2-bit	|	2-bit	|	PL	|	arm_convolve_HWC_u8_u2_u2.c	|
|	arm_convolve_HWC_u8_u2_u2_icn	|	8-bit	|	2-bit	|	2-bit	|	PL+ICN	|	arm_convolve_HWC_u8_u2_u2_icn.c	|
|	arm_convolve_HWC_u8_u2_u2_PACT_CH_icn	|	8-bit	|	2-bit	|	2-bit	|	PC+ICN	|	arm_convolve_HWC_u8_u2_u2_PACT_CH_icn.c	|
|	arm_convolve_HWC_u8_u2_u4	|	8-bit	|	2-bit	|	4-bit	|	PL	|	arm_convolve_HWC_u8_u2_u4.c	|
|	arm_convolve_HWC_u8_u2_u4_icn	|	8-bit	|	2-bit	|	4-bit	|	PL+ICN	|	arm_convolve_HWC_u8_u2_u4_icn.c	|
|	arm_convolve_HWC_u8_u2_u4_PACT_CH_icn	|	8-bit	|	2-bit	|	4-bit	|	PC+ICN	|	arm_convolve_HWC_u8_u2_u4_PACT_CH_icn.c	|
|	arm_convolve_HWC_u8_u2_u8	|	8-bit	|	2-bit	|	8-bit	|	PL	|	arm_convolve_HWC_u8_u2_u8.c	|
|	arm_convolve_HWC_u8_u2_u8_icn	|	8-bit	|	2-bit	|	8-bit	|	PL+ICN	|	arm_convolve_HWC_u8_u2_u8_icn.c	|
|	arm_convolve_HWC_u8_u2_u8_PACT_CH_icn	|	8-bit	|	2-bit	|	8-bit	|	PC+ICN	|	arm_convolve_HWC_u8_u2_u8_PACT_CH_icn.c	|
|	arm_convolve_HWC_u8_u4_u2	|	8-bit	|	4-bit	|	2-bit	|	PL	|	arm_convolve_HWC_u8_u4_u2.c	|
|	arm_convolve_HWC_u8_u4_u2_icn	|	8-bit	|	4-bit	|	2-bit	|	PL+ICN	|	arm_convolve_HWC_u8_u4_u2_icn.c	|
|	arm_convolve_HWC_u8_u4_u2_PACT_CH_icn	|	8-bit	|	4-bit	|	2-bit	|	PC+ICN	|	arm_convolve_HWC_u8_u4_u2_PACT_CH_icn.c	|
|	arm_convolve_HWC_u8_u4_u4	|	8-bit	|	4-bit	|	4-bit	|	PL	|	arm_convolve_HWC_u8_u4_u4.c	|
|	arm_convolve_HWC_u8_u4_u4_icn	|	8-bit	|	4-bit	|	4-bit	|	PL+ICN	|	arm_convolve_HWC_u8_u4_u4_icn.c	|
|	arm_convolve_HWC_u8_u4_u4_PACT_CH_icn	|	8-bit	|	4-bit	|	4-bit	|	PC+ICN	|	arm_convolve_HWC_u8_u4_u4_PACT_CH_icn.c	|
|	arm_convolve_HWC_u8_u4_u8	|	8-bit	|	4-bit	|	8-bit	|	PL	|	arm_convolve_HWC_u8_u4_u8.c	|
|	arm_convolve_HWC_u8_u4_u8_icn	|	8-bit	|	4-bit	|	8-bit	|	PL+ICN	|	arm_convolve_HWC_u8_u4_u8_icn.c	|
|	arm_convolve_HWC_u8_u4_u8_PACT_CH_icn	|	8-bit	|	4-bit	|	8-bit	|	PC+ICN	|	arm_convolve_HWC_u8_u4_u8_PACT_CH_icn.c	|
|	arm_convolve_HWC_u8_u8_u2	|	8-bit	|	8-bit	|	2-bit	|	PL	|	arm_convolve_HWC_u8_u8_u2.c	|
|	arm_convolve_HWC_u8_u8_u2_icn	|	8-bit	|	8-bit	|	2-bit	|	PL+ICN	|	arm_convolve_HWC_u8_u8_u2_icn.c	|
|	arm_convolve_HWC_u8_u8_u2_PACT_CH_icn	|	8-bit	|	8-bit	|	2-bit	|	PC+ICN	|	arm_convolve_HWC_u8_u8_u2_PACT_CH_icn.c	|
|	arm_convolve_HWC_u8_u8_u4	|	8-bit	|	8-bit	|	4-bit	|	PL	|	arm_convolve_HWC_u8_u8_u4.c	|
|	arm_convolve_HWC_u8_u8_u4_icn	|	8-bit	|	8-bit	|	4-bit	|	PL+ICN	|	arm_convolve_HWC_u8_u8_u4_icn.c	|
|	arm_convolve_HWC_u8_u8_u4_PACT_CH_icn	|	8-bit	|	8-bit	|	4-bit	|	PC+ICN	|	arm_convolve_HWC_u8_u8_u4_PACT_CH_icn.c	|
|	arm_convolve_HWC_u8_u8_u8	|	8-bit	|	8-bit	|	8-bit	|	PL	|	arm_convolve_HWC_u8_u8_u8.c	|
|	arm_convolve_HWC_u8_u8_u8_icn	|	8-bit	|	8-bit	|	8-bit	|	PL+ICN	|	arm_convolve_HWC_u8_u8_u8_icn.c	|
|	arm_convolve_HWC_u8_u8_u8_PACT_CH_icn	|	8-bit	|	8-bit	|	8-bit	|	PC+ICN	|	arm_convolve_HWC_u8_u8_u8_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u2_u2	|	2-bit	|	2-bit	|	2-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u2_u2_u2.c	|
|	arm_depthwise_separable_conv_HWC_u2_u2_u2_icn	|	2-bit	|	2-bit	|	2-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u2_u2_u2_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u2_u2_PACT_CH_icn	|	2-bit	|	2-bit	|	2-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u2_u2_u2_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u2_u4	|	2-bit	|	2-bit	|	4-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u2_u2_u4.c	|
|	arm_depthwise_separable_conv_HWC_u2_u2_u4_icn	|	2-bit	|	2-bit	|	4-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u2_u2_u4_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u2_u4_PACT_CH_icn	|	2-bit	|	2-bit	|	4-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u2_u2_u4_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u2_u8	|	2-bit	|	2-bit	|	8-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u2_u2_u8.c	|
|	arm_depthwise_separable_conv_HWC_u2_u2_u8_icn	|	2-bit	|	2-bit	|	8-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u2_u2_u8_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u2_u8_PACT_CH_icn	|	2-bit	|	2-bit	|	8-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u2_u2_u8_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u4_u2	|	2-bit	|	4-bit	|	2-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u2_u4_u2.c	|
|	arm_depthwise_separable_conv_HWC_u2_u4_u2_icn	|	2-bit	|	4-bit	|	2-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u2_u4_u2_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u4_u2_PACT_CH_icn	|	2-bit	|	4-bit	|	2-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u2_u4_u2_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u4_u4	|	2-bit	|	4-bit	|	4-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u2_u4_u4.c	|
|	arm_depthwise_separable_conv_HWC_u2_u4_u4_icn	|	2-bit	|	4-bit	|	4-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u2_u4_u4_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u4_u4_PACT_CH_icn	|	2-bit	|	4-bit	|	4-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u2_u4_u4_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u4_u8	|	2-bit	|	4-bit	|	8-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u2_u4_u8.c	|
|	arm_depthwise_separable_conv_HWC_u2_u4_u8_icn	|	2-bit	|	4-bit	|	8-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u2_u4_u8_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u4_u8_PACT_CH_icn	|	2-bit	|	4-bit	|	8-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u2_u4_u8_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u8_u2	|	2-bit	|	8-bit	|	2-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u2_u8_u2.c	|
|	arm_depthwise_separable_conv_HWC_u2_u8_u2_icn	|	2-bit	|	8-bit	|	2-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u2_u8_u2_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u8_u2_PACT_CH_icn	|	2-bit	|	8-bit	|	2-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u2_u8_u2_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u8_u4	|	2-bit	|	8-bit	|	4-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u2_u8_u4.c	|
|	arm_depthwise_separable_conv_HWC_u2_u8_u4_icn	|	2-bit	|	8-bit	|	4-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u2_u8_u4_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u8_u4_PACT_CH_icn	|	2-bit	|	8-bit	|	4-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u2_u8_u4_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u8_u8	|	2-bit	|	8-bit	|	8-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u2_u8_u8.c	|
|	arm_depthwise_separable_conv_HWC_u2_u8_u8_icn	|	2-bit	|	8-bit	|	8-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u2_u8_u8_icn.c	|
|	arm_depthwise_separable_conv_HWC_u2_u8_u8_PACT_CH_icn	|	2-bit	|	8-bit	|	8-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u2_u8_u8_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u2_u2	|	4-bit	|	2-bit	|	2-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u4_u2_u2.c	|
|	arm_depthwise_separable_conv_HWC_u4_u2_u2_icn	|	4-bit	|	2-bit	|	2-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u4_u2_u2_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u2_u2_PACT_CH_icn	|	4-bit	|	2-bit	|	2-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u4_u2_u2_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u2_u4	|	4-bit	|	2-bit	|	4-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u4_u2_u4.c	|
|	arm_depthwise_separable_conv_HWC_u4_u2_u4_icn	|	4-bit	|	2-bit	|	4-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u4_u2_u4_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u2_u4_PACT_CH_icn	|	4-bit	|	2-bit	|	4-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u4_u2_u4_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u2_u8	|	4-bit	|	2-bit	|	8-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u4_u2_u8.c	|
|	arm_depthwise_separable_conv_HWC_u4_u2_u8_icn	|	4-bit	|	2-bit	|	8-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u4_u2_u8_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u2_u8_PACT_CH_icn	|	4-bit	|	2-bit	|	8-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u4_u2_u8_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u4_u2	|	4-bit	|	4-bit	|	2-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u4_u4_u2.c	|
|	arm_depthwise_separable_conv_HWC_u4_u4_u2_icn	|	4-bit	|	4-bit	|	2-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u4_u4_u2_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u4_u2_PACT_CH_icn	|	4-bit	|	4-bit	|	2-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u4_u4_u2_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u4_u4	|	4-bit	|	4-bit	|	4-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u4_u4_u4.c	|
|	arm_depthwise_separable_conv_HWC_u4_u4_u4_icn	|	4-bit	|	4-bit	|	4-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u4_u4_u4_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u4_u4_PACT_CH_icn	|	4-bit	|	4-bit	|	4-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u4_u4_u4_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u4_u8	|	4-bit	|	4-bit	|	8-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u4_u4_u8.c	|
|	arm_depthwise_separable_conv_HWC_u4_u4_u8_icn	|	4-bit	|	4-bit	|	8-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u4_u4_u8_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u4_u8_PACT_CH_icn	|	4-bit	|	4-bit	|	8-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u4_u4_u8_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u8_u2	|	4-bit	|	8-bit	|	2-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u4_u8_u2.c	|
|	arm_depthwise_separable_conv_HWC_u4_u8_u2_icn	|	4-bit	|	8-bit	|	2-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u4_u8_u2_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u8_u2_PACT_CH_icn	|	4-bit	|	8-bit	|	2-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u4_u8_u2_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u8_u4	|	4-bit	|	8-bit	|	4-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u4_u8_u4.c	|
|	arm_depthwise_separable_conv_HWC_u4_u8_u4_icn	|	4-bit	|	8-bit	|	4-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u4_u8_u4_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u8_u4_PACT_CH_icn	|	4-bit	|	8-bit	|	4-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u4_u8_u4_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u8_u8	|	4-bit	|	8-bit	|	8-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u4_u8_u8.c	|
|	arm_depthwise_separable_conv_HWC_u4_u8_u8_icn	|	4-bit	|	8-bit	|	8-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u4_u8_u8_icn.c	|
|	arm_depthwise_separable_conv_HWC_u4_u8_u8_PACT_CH_icn	|	4-bit	|	8-bit	|	8-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u4_u8_u8_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u2_u2	|	8-bit	|	2-bit	|	2-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u8_u2_u2.c	|
|	arm_depthwise_separable_conv_HWC_u8_u2_u2_icn	|	8-bit	|	2-bit	|	2-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u8_u2_u2_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u2_u2_PACT_CH_icn	|	8-bit	|	2-bit	|	2-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u8_u2_u2_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u2_u4	|	8-bit	|	2-bit	|	4-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u8_u2_u4.c	|
|	arm_depthwise_separable_conv_HWC_u8_u2_u4_icn	|	8-bit	|	2-bit	|	4-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u8_u2_u4_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u2_u4_PACT_CH_icn	|	8-bit	|	2-bit	|	4-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u8_u2_u4_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u2_u8	|	8-bit	|	2-bit	|	8-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u8_u2_u8.c	|
|	arm_depthwise_separable_conv_HWC_u8_u2_u8_icn	|	8-bit	|	2-bit	|	8-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u8_u2_u8_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u2_u8_PACT_CH_icn	|	8-bit	|	2-bit	|	8-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u8_u2_u8_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u4_u2	|	8-bit	|	4-bit	|	2-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u8_u4_u2.c	|
|	arm_depthwise_separable_conv_HWC_u8_u4_u2_icn	|	8-bit	|	4-bit	|	2-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u8_u4_u2_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u4_u2_PACT_CH_icn	|	8-bit	|	4-bit	|	2-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u8_u4_u2_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u4_u4	|	8-bit	|	4-bit	|	4-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u8_u4_u4.c	|
|	arm_depthwise_separable_conv_HWC_u8_u4_u4_icn	|	8-bit	|	4-bit	|	4-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u8_u4_u4_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u4_u4_PACT_CH_icn	|	8-bit	|	4-bit	|	4-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u8_u4_u4_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u4_u8	|	8-bit	|	4-bit	|	8-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u8_u4_u8.c	|
|	arm_depthwise_separable_conv_HWC_u8_u4_u8_icn	|	8-bit	|	4-bit	|	8-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u8_u4_u8_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u4_u8_PACT_CH_icn	|	8-bit	|	4-bit	|	8-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u8_u4_u8_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u8_u2	|	8-bit	|	8-bit	|	2-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u8_u8_u2.c	|
|	arm_depthwise_separable_conv_HWC_u8_u8_u2_icn	|	8-bit	|	8-bit	|	2-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u8_u8_u2_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u8_u2_PACT_CH_icn	|	8-bit	|	8-bit	|	2-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u8_u8_u2_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u8_u4	|	8-bit	|	8-bit	|	4-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u8_u8_u4.c	|
|	arm_depthwise_separable_conv_HWC_u8_u8_u4_icn	|	8-bit	|	8-bit	|	4-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u8_u8_u4_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u8_u4_PACT_CH_icn	|	8-bit	|	8-bit	|	4-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u8_u8_u4_PACT_CH_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u8_u8	|	8-bit	|	8-bit	|	8-bit	|	PL	|	arm_depthwise_separable_conv_HWC_u8_u8_u8.c	|
|	arm_depthwise_separable_conv_HWC_u8_u8_u8_icn	|	8-bit	|	8-bit	|	8-bit	|	PL+ICN	|	arm_depthwise_separable_conv_HWC_u8_u8_u8_icn.c	|
|	arm_depthwise_separable_conv_HWC_u8_u8_u8_PACT_CH_icn	|	8-bit	|	8-bit	|	8-bit	|	PC+ICN	|	arm_depthwise_separable_conv_HWC_u8_u8_u8_PACT_CH_icn.c	|

