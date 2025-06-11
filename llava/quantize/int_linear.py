import torch
import torch.nn as nn
import torch.nn.functional as F
from llava.quantize.quantizer import UniformAffineQuantizer
from collections import defaultdict



class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        # self.register_buffer('weight',org_module.weight)
        self.register_parameter('weight', org_module.weight)
        if org_module.bias is not None:
            # self.register_buffer('bias',org_module.bias)
            self.register_parameter('bias', org_module.bias)
        else:
            self.bias = None

        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = True
        # weight_quant_params['lwc'] = True
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

        self.scale = None

    @staticmethod
    def compute_weighted_index_scores(topk_tensor: torch.Tensor, weight_scheme: str = 'linear') -> dict:
        """
        计算加权 index 得分，考虑出现频率和在 topk 中的位置。

        参数：
            topk_tensor (torch.Tensor): shape 为 (a, b)，a 是样本数，b 是每个样本的 top-k index。
            weight_scheme (str): 位置权重策略，可选：
                - 'linear'（默认）：线性权重，位置越靠前权重越高（如 3,2,1）
                - 'exp'：指数权重，前排衰减慢，后排衰减快

        返回：
            dict: {index: score}，按 index 聚合的加权得分
        """
        a, b = topk_tensor.shape

        # 定义位置权重
        if weight_scheme == 'linear':
            position_weights = torch.arange(b, 0, -1).float()
        elif weight_scheme == 'exp':
            position_weights = torch.exp(-0.5 * torch.arange(b).float())
        else:
            raise ValueError("Unsupported weight_scheme. Use 'linear' or 'exp'.")

        # 累加每个 index 的分数
        score_dict = defaultdict(float)
        for row in topk_tensor:
            for pos, index in enumerate(row):
                score_dict[int(index)] += float(position_weights[pos])

        return dict(score_dict)

    def forward(self, input_t: torch.Tensor):

        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        # elif self.use_weight_quant:
        #     weight = self.weight_quantizer(self.weight)
        #     bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input_t = self.act_quantizer(input_t, inplace=False)

        out = self.fwd_func(input_t, weight, bias, **self.fwd_kwargs)


        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
