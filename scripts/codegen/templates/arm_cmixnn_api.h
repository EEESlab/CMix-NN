% if config.api=="CMixNNConvolve":
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
% elif config.folding=="icn":
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
% else:
                        const uint8_t z_out,
                        const int32_t m_zero,
                        const int8_t n_zero,
% endif
                        int16_t * bufferA,
                        uint8_t * bufferB);
% elif config.api=="CMixNNMatMul":
 /**
   * @brief Matrix-Multiplication function for ${config.wt_data_t} x int16_t convolution with reordered columns.
   *        Output is then quantized to ${config.out_data_t} using ${config.folding} config.folding technique.
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @param[in]       z_a         A operand offset
% if config.quantization=="PACT":
   * @param[in]       z_a         A operand offset
% elif config.quantization=="PACT_CH":
   * @param[in]       *z_a        pointer to A operand offsets (per-output channel)
% endif
% if config.folding=="thr":
   * @param[in]       thresholds  pointer to thresholds for quantization
% elif config.folding=="icn":
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      pointer to m zero quantization params (per-output-ch)
   * @param[in]       n_zero      pointer to n zero quantization params (per-output-ch)
% else:
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
% endif
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *
${config.fn_name}(const uint8_t * pA,
                        const int16_t * pInBuffer,
                        const uint16_t ch_im_out,
                        const uint16_t numCol_A,
                        const int32_t * bias,
                        uint8_t * pOut,
% if config.quantization=="PACT":
                        const uint8_t z_a,
% elif config.quantization=="PACT_CH":
                        const uint8_t *z_a,
% endif
% if config.folding=="thr":
                        const int16_t * thresholds);
% elif config.folding=="icn":
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero);
% else:
                        const uint8_t z_out,
                        const int32_t m_zero,
                        const int8_t n_zero);
% endif
% elif config.api=="CMixNNDepthwise":
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
% elif config.folding=="icn":
                        const uint8_t z_out,
                        const int32_t *m_zero,
                        const int8_t *n_zero,
% else:
                        const uint8_t z_out,
                        const int32_t m_zero,
                        const int8_t n_zero,
% endif
                        int16_t * bufferA,
                        uint8_t * bufferB);
% elif config.api=="CMixNNConvertReorder":
/**
 * @brief Converts the elements of ${config.in_data_t} vector to
 *        a reordered ${config.out_data_t} vector (without left-shift).
 * @param[in] *pSrc       points to the ${config.in_data_t} input vector
 * @param[out] *pDst      points to the ${config.out_data_t} output vector
 * @param[in] blockSize   length of the input vector
 * @param[in] offset      input quantization offset
 * @return none.
 */

void
${config.fn_name}(
                        const uint8_t *pSrc,
                        int16_t *pDst,
                        uint32_t blockSize,
                        const uint8_t offset);
% endif
