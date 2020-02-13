import errno
import os
from mako.template import Template


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


CMixNNInstallPath = cwd = os.getcwd() + "/../../"
CMixNNSrcDirs = {'Include': CMixNNInstallPath + "Include",
                'convolution': CMixNNInstallPath + "Source/ConvolutionFunctions/",
                'fullyConnected': CMixNNInstallPath + "Source/FullyConnectedFunctions/",
                'Pooling': CMixNNInstallPath + "Source/PoolingFunctions/",
                'NNSupport': CMixNNInstallPath + "Source/NNSupportFunctions/"}
CMixNNDataPrecisions = ['u8', 'u4', 'u2']
CMixNNQuantizationMethods = ['PACT', 'PACT_CH']
CMixNNFoldingMethods = ['weights', 'icn'] 
CMixNNConstrains = {'u8': 4, 'u4': 8, 'u2': 16}
CMixNNAPI = "\n"
CMixNNSupportAPI = "\n"


class CMixNNFactory(object):
    def __init__(self, in_data_t, out_data_t, wt_data_t):
        self.in_data_t = in_data_t
        self.out_data_t = out_data_t
        self.wt_data_t = wt_data_t
        self.arithmetic_t = 'int16'
        self.fn_name = ''
        self.filename = ''
        self.quantization = ''
        self.folding = ''
        self.api = ''
        self.header = ''

    def generate_api(self):
        return Template(filename="templates/arm_cmixnn_api.h").render(config=self)

    def generate_header(self):
        return Template(filename="templates/arm_cmixnn_header.h").render(config=self)
        


class CMixNNConvolve(CMixNNFactory):
    def __init__(self, in_data_t, out_data_t, wt_data_t, quantization, folding):
        super().__init__(in_data_t, out_data_t, wt_data_t)
        self.fn_name = "arm_convolve_HWC_{0}_{1}_{2}{3}{4}".format(str(in_data_t), str(out_data_t), str(wt_data_t), str(
            "_" + quantization if quantization != "PACT" else ""), str("_" + folding if folding != "weights" else ""))
        self.filename = self.fn_name + ".c"
        self.quantization = quantization
        self.folding = folding
        self.reordered_no_shift_load_fn = "arm_{0}_to_{1}_reordered".format(str(
            self.in_data_t), self.arithmetic_t)
        self.nn_mat_mul_fn = "arm_nn_mat_mult_kernel_reordered_{0}_{1}_{2}{3}{4}".format(str(wt_data_t),
                                                                                         str(self.arithmetic_t),
                                                                                         str(out_data_t), str(
                "_" + quantization if quantization != "PACT" else ""), str(
                "_" + folding if folding != "weights" else ""))
        self.ch_in_constrain = CMixNNConstrains[in_data_t]
        self.ch_out_constrain = CMixNNConstrains[out_data_t]
        self.api = self.__class__.__name__

    def generate_code(self):
        self.header = self.generate_header()
        return Template(filename="templates/arm_convolve_HWC_x_y_z.c").render(config=self)

    def get_leftover_code(self):
        return ""


class CMixNNDepthwise(CMixNNFactory):
    def __init__(self, in_data_t, out_data_t, wt_data_t, quantization, folding):
        super().__init__(in_data_t, out_data_t, wt_data_t)
        self.fn_name = "arm_depthwise_separable_conv_HWC_{0}_{1}_{2}{3}{4}".format(str(in_data_t), str(out_data_t), str(wt_data_t), str(
            "_" + quantization if quantization != "PACT" else ""), str("_" + folding if folding != "weights" else ""))
        self.filename = self.fn_name + ".c"
        self.quantization = quantization
        self.folding = folding
        self.api = self.__class__.__name__

    def generate_code(self):
        self.header = self.generate_header()
        return Template(filename="templates/arm_depthwise_separable_conv_HWC_x_y_z.c").render(config=self)


class CMixNNMatMul(CMixNNFactory):
    def __init__(self, out_data_t, wt_data_t, quantization, folding):
        super().__init__("", out_data_t, wt_data_t)
        self.fn_name = "arm_nn_mat_mult_kernel_reordered_{0}_{1}_{2}{3}{4}".format(str(wt_data_t),
                                                                                   str(self.arithmetic_t),
                                                                                   str(out_data_t), str(
                "_" + quantization if quantization != "PACT" else ""), str(
                "_" + folding if folding != "weights" else ""))
        self.filename = self.fn_name + ".c"
        self.quantization = quantization
        self.folding = folding
        self.api = self.__class__.__name__

    def generate_code(self):
        self.header = self.generate_header()
        return Template(filename="templates/arm_nn_mat_mult_kernel_reordered_x_y_z.c").render(config=self)


class CMixNNConvertReorder(CMixNNFactory):
    def __init__(self, in_data_t):
        super().__init__(in_data_t, "", "")
        self.fn_name = "arm_{0}_to_{1}_reordered".format(str(in_data_t), str(self.arithmetic_t))
        self.filename = self.fn_name + ".c"
        self.out_data_t = str(self.arithmetic_t)
        self.api = self.__class__.__name__

    def generate_code(self):
        self.header = self.generate_header()
        return Template(filename="templates/arm_x_to_y_reordered.c").render(config=self)


# Generate CMixNNConvolve
mkdir_p(CMixNNSrcDirs['convolution'])
for i in CMixNNDataPrecisions:
    for j in CMixNNDataPrecisions:
        for z in CMixNNDataPrecisions:
            for q in CMixNNQuantizationMethods:
                for f in CMixNNFoldingMethods:
                    if (q == "PACT_CH" and f != "weights") or q == "PACT":
                        c = CMixNNConvolve(in_data_t=i, out_data_t=j, wt_data_t=z, quantization=q, folding=f)
                        CMixNNAPI += c.generate_api() + "\n"
                        new_file = open(CMixNNSrcDirs['convolution'] + c.filename, 'w')
                        new_file.write(c.generate_code())
                        new_file.close()

# Generate CMixNNDepthwise
mkdir_p(CMixNNSrcDirs['convolution'])
for i in CMixNNDataPrecisions:
    for j in CMixNNDataPrecisions:
        for z in CMixNNDataPrecisions:
            for q in CMixNNQuantizationMethods:
                for f in CMixNNFoldingMethods:
                    if (q == "PACT_CH" and f != "weights") or q == "PACT":
                        c = CMixNNDepthwise(in_data_t=i, out_data_t=j, wt_data_t=z, quantization=q, folding=f)
                        CMixNNAPI += c.generate_api() + "\n"
                        new_file = open(CMixNNSrcDirs['convolution'] + c.filename, 'w')
                        new_file.write(c.generate_code())
                        new_file.close()

# Generate CMixNNMatMul
mkdir_p(CMixNNSrcDirs['convolution'])
for i in CMixNNDataPrecisions:
    for j in CMixNNDataPrecisions:
        for q in CMixNNQuantizationMethods:
            for f in CMixNNFoldingMethods:
                if (q == "PACT_CH" and f != "weights") or q == "PACT":
                    c = CMixNNMatMul(out_data_t=i, wt_data_t=j, quantization=q, folding=f)
                    CMixNNAPI += c.generate_api() + "\n"
                    new_file = open(CMixNNSrcDirs['convolution'] + c.filename, 'w')
                    new_file.write(c.generate_code())
                    new_file.close()

# Generate CMixNNConvertReorder
mkdir_p(CMixNNSrcDirs['NNSupport'])
for i in CMixNNDataPrecisions:
    c = CMixNNConvertReorder(in_data_t=i)
    CMixNNSupportAPI += c.generate_api() + "\n"
    new_file = open(CMixNNSrcDirs['NNSupport'] + c.filename, 'w')
    new_file.write(c.generate_code())
    new_file.close()

# Generate new include files
mkdir_p(CMixNNSrcDirs['Include'])
new_file = open(CMixNNSrcDirs['Include'] + "/arm_cmixnn.h", 'w')
new_file.write(Template(filename="templates/arm_cmixnn.h").render(CMixNNAPI=CMixNNAPI))
new_file.close()
new_file = open(CMixNNSrcDirs['Include'] + "/arm_cmixnn_support.h", 'w')
new_file.write(Template(filename="templates/arm_cmixnn_support.h").render(CMixNNSupportAPI=CMixNNSupportAPI))
new_file.close()
