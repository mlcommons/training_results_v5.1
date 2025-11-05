from torch import nn


def linear_weight_init(layer):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.constant_(layer.bias, 0)


def mlpembedder_weight_init(module, init_std: float = 0.02):
    nn.init.normal_(module.in_layer.weight, std=init_std)
    nn.init.constant_(module.in_layer.bias, 0)
    nn.init.normal_(module.out_layer.weight, std=init_std)
    nn.init.constant_(module.out_layer.bias, 0)


def mlp_weight_init(layer):
    linear_weight_init(layer.linear_fc1)
    linear_weight_init(layer.linear_fc2)


def attention_weight_init(module):
    linear_weight_init(module.linear_proj)
    linear_weight_init(module.linear_qkv)
    module.q_layernorm.reset_parameters()
    module.k_layernorm.reset_parameters()


def adaLN_modulation_weight_init(module, with_second_ln=False):
    module.ln.reset_parameters()
    if with_second_ln:
        module.ln2.reset_parameters()
    nn.init.constant_(module.adaLN_modulation[-1].weight, 0)
    nn.init.constant_(module.adaLN_modulation[-1].bias, 0)


def init_single_block_weights(module):
    # commented out ones are handled by megatron through transformer_config
    #attention_weight_init(module.self_attention)
    #mlp_weight_init(module.mlp)
    adaLN_modulation_weight_init(module.adaln)


def joint_attention_weight_init(module):
    linear_weight_init(module.linear_proj)
    linear_weight_init(module.added_linear_proj)
    linear_weight_init(module.linear_qkv)
    linear_weight_init(module.added_linear_qkv)
    module.q_layernorm.reset_parameters()
    module.added_q_layernorm.reset_parameters()
    module.k_layernorm.reset_parameters()
    module.added_k_layernorm.reset_parameters()


def init_double_block_weights(module):
    # commented out ones are handled by megatron through transformer_config
    #mlp_weight_init(module.mlp)
    #mlp_weight_init(module.context_mlp)
    adaLN_modulation_weight_init(module.adaln, with_second_ln=True)
    adaLN_modulation_weight_init(module.adaln_context, with_second_ln=True)
    #joint_attention_weight_init(module.self_attention)
