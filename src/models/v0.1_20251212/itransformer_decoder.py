#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
iTransformer Decoder-Only 主模型
版本: v0.1
日期: 20251212

实现高度可配置的iTransformer模型，支持ResNet风格的全连接层模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
import importlib.util

# 动态导入DecoderLayer
def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 导入DecoderLayer
models_path = Path(__file__).parent
try:
    decoder_layer_module = _load_module(models_path / "decoder_layer.py", "decoder_layer")
    DecoderLayer = decoder_layer_module.DecoderLayer
except Exception as e:
    raise ImportError(f"Failed to import DecoderLayer: {e}") from e


class ResidualFCBlock(nn.Module):
    """
    ResNet风格的全连接残差块
    支持多层堆叠，每层都有残差连接
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        activation: str = "gelu",
        use_bias: bool = True
    ):
        """
        初始化ResNet风格的FC残差块
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表（每层的输出维度）
            dropout: Dropout比率
            activation: 激活函数，"gelu"或"relu"
            use_bias: 是否使用bias
        """
        super(ResidualFCBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # 激活函数
        if activation.lower() == "gelu":
            act_fn = F.gelu
        elif activation.lower() == "relu":
            act_fn = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 构建层
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            # Linear层
            self.layers.append(nn.Linear(prev_dim, hidden_dim, bias=use_bias))
            
            # Dropout
            self.dropouts.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        self.activation = act_fn
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim
        
        # 如果输入输出维度不同，创建残差投影层
        if self.input_dim != self.output_dim:
            self.residual_proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, ..., input_dim)
        
        Returns:
            输出张量，形状为 (batch_size, ..., output_dim)
        """
        residual = x
        
        # 如果输入输出维度不同，需要投影残差
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        # 通过所有层（无归一化）
        out = x
        for linear, dropout in zip(self.layers, self.dropouts):
            out = linear(out)
            out = self.activation(out)
            out = dropout(out)
        
        # 残差连接
        out = out + residual
        
        return out


class ResidualFCBlock5Layer(nn.Module):
    """
    固定5层的ResNet风格全连接残差块
    所有5层使用相同的维度，简化配置
    """
    
    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_bias: bool = True
    ):
        """
        初始化固定5层的ResNet FC残差块
        
        Args:
            dim: 统一的维度（输入、输出和所有中间层都是这个维度）
            dropout: Dropout比率
            activation: 激活函数，"gelu"或"relu"
            use_bias: 是否使用bias
        """
        super(ResidualFCBlock5Layer, self).__init__()
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = F.gelu
        elif activation.lower() == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 固定5层，每层都是 dim -> dim
        self.layer1 = nn.Linear(dim, dim, bias=use_bias)
        self.layer2 = nn.Linear(dim, dim, bias=use_bias)
        self.layer3 = nn.Linear(dim, dim, bias=use_bias)
        self.layer4 = nn.Linear(dim, dim, bias=use_bias)
        self.layer5 = nn.Linear(dim, dim, bias=use_bias)
        
        # 只在最后一层之后、残差连接之前使用一个Dropout
        # 移除前4层的Dropout，避免过度正则化
        self.dropout = nn.Dropout(dropout)
        
        self.input_dim = dim
        self.output_dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, ..., dim)
        
        Returns:
            输出张量，形状为 (batch_size, ..., dim)
        """
        residual = x
        
        # 通过5层（前4层不使用Dropout，只在最后一层之后使用）
        out = self.layer1(x)
        out = self.activation(out)
        # 第1层：不使用Dropout
        
        out = self.layer2(out)
        out = self.activation(out)
        # 第2层：不使用Dropout
        
        out = self.layer3(out)
        out = self.activation(out)
        # 第3层：不使用Dropout
        
        out = self.layer4(out)
        out = self.activation(out)
        # 第4层：不使用Dropout
        
        out = self.layer5(out)
        out = self.activation(out)
        # 第5层：在激活函数之后、残差连接之前使用Dropout
        out = self.dropout(out)
        
        # 残差连接（输入输出维度相同，直接相加）
        out = out + residual
        
        return out


class ResNetFCModule(nn.Module):
    """
    ResNet风格的全连接模块
    可以包含多个ResidualFCBlock
    """
    
    def __init__(
        self,
        input_dim: int,
        modules_config: List[Dict[str, Any]],
        dropout: float = 0.1,
        activation: str = "gelu",
        use_bias: bool = True
    ):
        """
        初始化ResNet FC模块
        
        Args:
            input_dim: 输入维度
            modules_config: 模块配置列表，每个元素包含：
                - name: 模块名称
                - layers: 层数
                - hidden_dims: 隐藏维度列表
                - dropout: Dropout比率（可选，使用默认值）
                - activation: 激活函数（可选，使用默认值）
                - use_bias: 是否使用bias（可选，使用默认值）
            dropout: 默认Dropout比率
            activation: 默认激活函数
            use_bias: 默认是否使用bias
        """
        super(ResNetFCModule, self).__init__()
        
        self.blocks = nn.ModuleList()
        
        prev_dim = input_dim
        for module_config in modules_config:
            # 获取配置参数
            module_name = module_config.get("name", "resnet_block")
            layers = module_config.get("layers", 1)
            hidden_dims = module_config.get("hidden_dims", [prev_dim])
            module_dropout = module_config.get("dropout", dropout)
            module_activation = module_config.get("activation", activation)
            module_use_bias = module_config.get("use_bias", use_bias)
            
            # 验证hidden_dims长度
            if len(hidden_dims) != layers:
                raise ValueError(
                    f"Module {module_name}: hidden_dims length ({len(hidden_dims)}) "
                    f"must equal layers ({layers})"
                )
            
            # 创建ResidualFCBlock
            block = ResidualFCBlock(
                input_dim=prev_dim,
                hidden_dims=hidden_dims,
                dropout=module_dropout,
                activation=module_activation,
                use_bias=module_use_bias
            )
            self.blocks.append(block)
            
            prev_dim = hidden_dims[-1]
        
        self.input_dim = input_dim
        self.output_dim = prev_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            输出张量
        """
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class iTransformerDecoder(nn.Module):
    """
    iTransformer Decoder-Only模型
    支持输入和输出的ResNet风格全连接层模块
    """
    
    def __init__(
        self,
        input_features: int,
        seq_len: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        decoder_config: Optional[Dict] = None,
        input_resnet_config: Optional[Dict] = None,
        time_aggregation_resnet_config: Optional[Dict] = None,
        output_resnet_config: Optional[Dict] = None,
        final_output_config: Optional[Dict] = None
    ):
        """
        初始化iTransformer Decoder模型
        
        Args:
            input_features: 输入特征数量
            seq_len: 输入序列长度
            d_model: 模型隐藏维度
            n_layers: Decoder层数
            n_heads: 注意力头数
            d_ff: Feed-Forward网络隐藏维度
            dropout: 默认Dropout比率
            activation: 默认激活函数
            decoder_config: Decoder层配置字典
            input_resnet_config: 输入ResNet模块配置字典
            time_aggregation_resnet_config: 时间维度聚合ResNet模块配置字典
            output_resnet_config: 输出ResNet模块配置字典
            final_output_config: 最终输出层配置字典
        """
        super(iTransformerDecoder, self).__init__()
        
        self.input_features = input_features
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 默认配置
        decoder_config = decoder_config or {}
        input_resnet_config = input_resnet_config or {}
        time_aggregation_resnet_config = time_aggregation_resnet_config or {}
        output_resnet_config = output_resnet_config or {}
        final_output_config = final_output_config or {}
        
        # ========== 输入ResNet模块 ==========
        self.input_resnet_enabled = input_resnet_config.get("enabled", False)
        if self.input_resnet_enabled:
            # 使用5层ResNet模块：输入维度 -> 128 -> 256 -> 512
            input_dims = input_resnet_config.get("dims", [128, 256, 512])
            input_dropout = input_resnet_config.get("dropout", dropout)
            input_activation = input_resnet_config.get("activation", activation)
            input_use_bias = input_resnet_config.get("use_bias", True)
            
            self.input_resnet_blocks = nn.ModuleList()
            prev_dim = input_features
            
            for dim in input_dims:
                # 创建5层ResNet块，维度从prev_dim转换到dim
                # 如果维度不同，需要先投影，然后应用5层ResNet
                if prev_dim != dim:
                    # 先投影到目标维度，添加激活函数
                    if input_activation.lower() == "gelu":
                        activation_layer = nn.GELU()
                    elif input_activation.lower() == "relu":
                        activation_layer = nn.ReLU()
                    else:
                        raise ValueError(f"Unsupported activation: {input_activation}")
                    projection = nn.Linear(prev_dim, dim, bias=input_use_bias)
                    # 然后应用5层ResNet（在目标维度上）
                    resnet_block = ResidualFCBlock5Layer(
                        dim=dim,
                        dropout=input_dropout,
                        activation=input_activation,
                        use_bias=input_use_bias
                    )
                    # 组合成Sequential：投影 -> 激活 -> ResNet块
                    block = nn.Sequential(projection, activation_layer, resnet_block)
                else:
                    # 直接应用5层ResNet
                    block = ResidualFCBlock5Layer(
                        dim=dim,
                        dropout=input_dropout,
                        activation=input_activation,
                        use_bias=input_use_bias
                    )
                self.input_resnet_blocks.append(block)
                prev_dim = dim
            
            input_to_decoder_dim = input_dims[-1] if input_dims else input_features
        else:
            self.input_resnet_blocks = None
            input_to_decoder_dim = input_features
        
        # ========== 输入Embedding层 ==========
        # 输入Embedding：维度变化需要激活函数
        if activation.lower() == "gelu":
            input_embedding_activation = nn.GELU()
        elif activation.lower() == "relu":
            input_embedding_activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.input_embedding = nn.Sequential(
            nn.Linear(input_to_decoder_dim, d_model),
            input_embedding_activation
        )
        
        # ========== Decoder层 ==========
        attn_embed_dim = decoder_config.get("attn_embed_dim", self.seq_len)
        decoder_layers = []
        for _ in range(n_layers):
            decoder_layer = DecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                use_causal_mask=decoder_config.get("use_causal_mask", True),
                norm_type=decoder_config.get("norm_type", "pre"),
                attention_dropout=decoder_config.get("attention_dropout", None),
                ff_dropout=decoder_config.get("ff_dropout", None),
                use_bias=decoder_config.get("use_bias", True),
                attn_embed_dim=attn_embed_dim
            )
            decoder_layers.append(decoder_layer)
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
        # ========== 时间维度聚合ResNet模块 ==========
        self.time_aggregation_resnet_enabled = time_aggregation_resnet_config.get("enabled", False)
        if self.time_aggregation_resnet_enabled:
            time_dims = time_aggregation_resnet_config.get("dims", [50])
            time_dropout = time_aggregation_resnet_config.get("dropout", dropout)
            time_activation = time_aggregation_resnet_config.get("activation", activation)
            time_use_bias = time_aggregation_resnet_config.get("use_bias", True)
            
            # 第一个ResNet块：100 -> 50（使用ResNet，有残差）
            self.time_aggregation_resnet_blocks = nn.ModuleList()
            prev_dim = seq_len
            
            for dim in time_dims:
                # 创建5层ResNet块，维度从prev_dim转换到dim
                if prev_dim != dim:
                    # 先投影到目标维度，添加激活函数
                    if time_activation.lower() == "gelu":
                        activation_layer = nn.GELU()
                    elif time_activation.lower() == "relu":
                        activation_layer = nn.ReLU()
                    else:
                        raise ValueError(f"Unsupported activation: {time_activation}")
                    projection = nn.Linear(prev_dim, dim, bias=time_use_bias)
                    # 然后应用5层ResNet（在目标维度上）
                    resnet_block = ResidualFCBlock5Layer(
                        dim=dim,
                        dropout=time_dropout,
                        activation=time_activation,
                        use_bias=time_use_bias
                    )
                    # 组合成Sequential：投影 -> 激活 -> ResNet块
                    block = nn.Sequential(projection, activation_layer, resnet_block)
                else:
                    # 直接应用5层ResNet
                    block = ResidualFCBlock5Layer(
                        dim=dim,
                        dropout=time_dropout,
                        activation=time_activation,
                        use_bias=time_use_bias
                    )
                self.time_aggregation_resnet_blocks.append(block)
                prev_dim = dim
            
            # 第二个压缩模块：50 -> 1（使用3层全连接，无残差，无dropout）
            # 50 -> 25 -> 5 -> 1
            self.time_aggregation_fc = nn.Sequential(
                nn.Linear(50, 25, bias=time_use_bias),
                nn.GELU() if time_activation.lower() == "gelu" else nn.ReLU(),
                nn.Linear(25, 5, bias=time_use_bias),
                nn.GELU() if time_activation.lower() == "gelu" else nn.ReLU(),
                nn.Linear(5, 1, bias=time_use_bias)
                # 最后一层无激活函数，直接输出
            )
        else:
            self.time_aggregation_resnet_blocks = None
            self.time_aggregation_fc = None
        
        # ========== 输出ResNet模块 ==========
        self.output_resnet_enabled = output_resnet_config.get("enabled", False)
        if self.output_resnet_enabled:
            # 使用5层ResNet模块：512 -> 256 -> 128 -> 64
            output_dims = output_resnet_config.get("dims", [256, 128, 64])
            output_dropout = output_resnet_config.get("dropout", dropout)
            output_activation = output_resnet_config.get("activation", activation)
            output_use_bias = output_resnet_config.get("use_bias", True)
            
            self.output_resnet_blocks = nn.ModuleList()
            prev_dim = d_model
            
            for dim in output_dims:
                # 创建5层ResNet块，维度从prev_dim转换到dim
                if prev_dim != dim:
                    # 先投影到目标维度，添加激活函数
                    if output_activation.lower() == "gelu":
                        activation_layer = nn.GELU()
                    elif output_activation.lower() == "relu":
                        activation_layer = nn.ReLU()
                    else:
                        raise ValueError(f"Unsupported activation: {output_activation}")
                    projection = nn.Linear(prev_dim, dim, bias=output_use_bias)
                    # 然后应用5层ResNet（在目标维度上）
                    resnet_block = ResidualFCBlock5Layer(
                        dim=dim,
                        dropout=output_dropout,
                        activation=output_activation,
                        use_bias=output_use_bias
                    )
                    # 组合成Sequential：投影 -> 激活 -> ResNet块
                    block = nn.Sequential(projection, activation_layer, resnet_block)
                else:
                    # 直接应用5层ResNet
                    block = ResidualFCBlock5Layer(
                        dim=dim,
                        dropout=output_dropout,
                        activation=output_activation,
                        use_bias=output_use_bias
                    )
                self.output_resnet_blocks.append(block)
                prev_dim = dim
            
            decoder_to_output_dim = output_dims[-1] if output_dims else d_model
        else:
            self.output_resnet_blocks = None
            decoder_to_output_dim = d_model
        
        # ========== 最终输出层 ==========
        # 使用3层全连接：64 -> 32 -> 16 -> 1
        # 前2层有激活函数但无dropout，最后1层无激活函数
        final_output_dim = final_output_config.get("output_dim", 1)
        final_activation = final_output_config.get("activation", activation)
        final_use_bias = final_output_config.get("use_bias", True)
        
        self.final_output = nn.Sequential(
            nn.Linear(decoder_to_output_dim, 32, bias=final_use_bias),
            nn.GELU() if final_activation.lower() == "gelu" else nn.ReLU(),
            # 第1层：无dropout
            nn.Linear(32, 16, bias=final_use_bias),
            nn.GELU() if final_activation.lower() == "gelu" else nn.ReLU(),
            # 第2层：无dropout
            nn.Linear(16, final_output_dim, bias=final_use_bias)
            # 第3层：无激活函数，直接输出
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_features)
        
        Returns:
            输出张量，形状为 (batch_size, output_dim)
        """
        batch_size, seq_len, input_features = x.shape
        
        # ========== 输入ResNet模块 ==========
        if self.input_resnet_enabled and self.input_resnet_blocks is not None:
            # Reshape: (batch_size, seq_len, input_features) -> (batch_size * seq_len, input_features)
            x_reshaped = x.view(-1, input_features)
            # 依次通过所有输入ResNet块
            for block in self.input_resnet_blocks:
                x_reshaped = block(x_reshaped)
            # Reshape back: (batch_size * seq_len, output_dim) -> (batch_size, seq_len, output_dim)
            x = x_reshaped.view(batch_size, seq_len, -1)
        
        # ========== 输入Embedding ==========
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)
        
        # ========== 转置：实现真正的iTransformer架构 ==========
        # 转置后，注意力在特征维度（d_model）上进行，而不是在时间维度（seq_len）上
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        
        # ========== Decoder层（注意力在d_model维度，即特征之间） ==========
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)  # (batch_size, d_model, seq_len)
        
        # ========== 转置回来 ==========
        x = x.transpose(1, 2)  # (batch_size, seq_len, d_model) = (32, 100, 512)
        
        # ========== 时间维度聚合ResNet模块 ==========
        if self.time_aggregation_resnet_enabled and self.time_aggregation_resnet_blocks is not None:
            # 转置以便在时间维度上操作: (batch, seq_len, d_model) -> (batch, d_model, seq_len)
            x = x.transpose(1, 2)  # (batch_size, d_model, seq_len) = (32, 512, 100)
            
            # Reshape: 将每个特征维度的时间序列展平
            batch_size, d_model, seq_len = x.shape
            x_reshaped = x.view(batch_size * d_model, seq_len)  # (16384, 100)
            
            # 依次通过所有时间维度聚合ResNet块（100 -> 50）
            for block in self.time_aggregation_resnet_blocks:
                x_reshaped = block(x_reshaped)  # (16384, 100) -> (16384, 50)
            
            # 通过3层全连接网络（50 -> 25 -> 5 -> 1，无残差，无dropout）
            x_reshaped = self.time_aggregation_fc(x_reshaped)  # (16384, 50) -> (16384, 1)
            
            # Reshape回原始结构: (batch * d_model, 1) -> (batch, d_model, 1)
            x = x_reshaped.view(batch_size, d_model, 1)  # (32, 512, 1)
            
            # 转置回 (batch, 1, d_model) 然后squeeze
            x = x.transpose(1, 2)  # (32, 1, 512)
            x = x.squeeze(1)  # (32, 512)
        else:
            # 如果没有启用时间维度聚合，直接取最后一个时间步
            x = x[:, -1, :]  # (batch_size, d_model)
        
        # ========== 输出ResNet模块 ==========
        if self.output_resnet_enabled and self.output_resnet_blocks is not None:
            # 依次通过所有输出ResNet块
            for block in self.output_resnet_blocks:
                x = block(x)  # (batch_size, output_resnet_output_dim)
        
        # ========== 最终输出层 ==========
        x = self.final_output(x)  # (batch_size, output_dim)
        
        return x
    
    @classmethod
    def from_config(cls, config_path: str) -> 'iTransformerDecoder':
        """
        从配置文件创建模型
        
        Args:
            config_path: 配置文件路径
        
        Returns:
            模型实例
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        model_config = config['model']
        
        return cls(
            input_features=model_config['input_features'],
            seq_len=model_config['seq_len'],
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            d_ff=model_config['d_ff'],
            dropout=model_config['dropout'],
            activation=model_config['activation'],
            decoder_config=model_config.get('decoder', {}),
            input_resnet_config=model_config.get('input_resnet', {}),
            time_aggregation_resnet_config=model_config.get('time_aggregation_resnet', {}),
            output_resnet_config=model_config.get('output_resnet', {}),
            final_output_config=model_config.get('final_output', {})
        )
    
    def get_num_parameters(self) -> int:
        """返回模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
