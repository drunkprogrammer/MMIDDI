import torch
import torch.nn as nn


class IntraDrugMMIMFM(nn.Module):
    def __init__(self, input_dim: int = 1024, num_heads: int = 4, num_layers: int = 2, drop_rate: float = 0.1):
        super(IntraDrugMMIMFM, self).__init__()
        self.input_dim = input_dim
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=input_dim * 4,
            batch_first=True, dropout=drop_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.self_attention = SelfAttention(input_dim, num_heads=num_heads, dropout=drop_rate)

    def forward(self, *modalities):
        drug1_vecs = [m[:, :self.input_dim] for m in modalities]
        drug2_vecs = [m[:, self.input_dim:] for m in modalities]
        x1 = torch.stack(drug1_vecs, dim=1)  # (batch, num_modalities, input_dim)
        x2 = torch.stack(drug2_vecs, dim=1)
        x1_encoded = self.transformer_encoder(x1)
        x2_encoded = self.transformer_encoder(x2)
        x1_attn_encoded, w_self1 = self.self_attention(x1_encoded)
        x2_attn_encoded, w_self2 = self.self_attention(x2_encoded)
        self_maps = {'self_d1': w_self1, 'self_d2': w_self2}
        return x1_attn_encoded, x2_attn_encoded, self_maps


class InterDrugMMIMFM(nn.Module):
    def __init__(self, input_dim: int = 1024, num_heads: int = 4, drop_rate: float = 0.1):
        super(InterDrugMMIMFM, self).__init__()
        self.input_dim = input_dim
        self.cross_attention = CrossAttention(input_dim, num_heads=num_heads, dropout=drop_rate)

    def forward(self, x1_attn_encoded, x2_attn_encoded):
        x1_cross_encoded, w_cross1 = self.cross_attention(x1_attn_encoded, x2_attn_encoded)
        x2_cross_encoded, w_cross2 = self.cross_attention(x2_attn_encoded, x1_attn_encoded)
        fused1 = x1_cross_encoded.contiguous().view(x1_cross_encoded.size(0), -1)
        fused2 = x2_cross_encoded.contiguous().view(x2_cross_encoded.size(0), -1)
        cross_maps = {'cross_d2_to_d1': w_cross1, 'cross_d1_to_d2': w_cross2}
        return fused1, fused2, cross_maps


class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        attn_output, attn_weights = self.attn(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        return x, attn_weights


class CrossAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        # query attends to context: Q=query, K=context, V=context
        # query/context: (batch, seq_len, input_dim)
        attn_output, attn_weights = self.attn(query, context, context)
        output = self.norm(query + self.dropout(attn_output))
        return output, attn_weights


class MMIMFM(nn.Module):

    """Multimodal Mutual Influence Mining & Fusion (MMIMFM)"""

    def __init__(self, model_dim: int = 64, num_heads: int = 4, num_layers: int = 1, drop_rate: float = 0.1):
        super(MMIMFM, self).__init__()
        self.model_dim = model_dim
        self.intra_module = IntraDrugMMIMFM(input_dim=model_dim, num_heads=num_heads, num_layers=num_layers, drop_rate=drop_rate)
        self.inter_module = InterDrugMMIMFM(input_dim=model_dim, num_heads=num_heads, drop_rate=drop_rate)

    def _unpack_modalities(self, modalities):
        """Return x1, x2 of shape (batch, num_modalities, model_dim).

        Accepts two calling styles:
         - modalities is a tuple of two tensors (d1, d2) already of shape (B, M, D)
         - modalities is a tuple/list of N modality tensors each shaped (B, 2*D)
        """
        if len(modalities) == 2 and isinstance(modalities[0], torch.Tensor) and modalities[0].dim() == 3:
            x1, x2 = modalities[0], modalities[1]
            return x1, x2

        drug1_vecs = [m[:, :self.model_dim] for m in modalities]
        drug2_vecs = [m[:, self.model_dim:] for m in modalities]
        x1 = torch.stack(drug1_vecs, dim=1)
        x2 = torch.stack(drug2_vecs, dim=1)
        return x1, x2

    def forward(self, *modalities):

        attn_maps = {
            'self_d1': None,
            'self_d2': None,
            'cross_d2_to_d1': None,
            'cross_d1_to_d2': None
        }

        x1_attn, x2_attn, self_maps = self.intra_module(*modalities)
        attn_maps['self_d1'] = self_maps.get('self_d1', None)
        attn_maps['self_d2'] = self_maps.get('self_d2', None)
        fused1, fused2, cross_maps = self.inter_module(x1_attn, x2_attn)
        attn_maps['cross_d2_to_d1'] = cross_maps.get('cross_d2_to_d1', None)
        attn_maps['cross_d1_to_d2'] = cross_maps.get('cross_d1_to_d2', None)
        return fused1, fused2, attn_maps


class Predictor(nn.Module):
    def __init__(self, drug_channels, drop_rate, num_modality, output_dim):
        super(Predictor, self).__init__()

        self.lstm1 = nn.LSTM(input_size=drug_channels * num_modality, hidden_size=drug_channels * 3, batch_first=True, dropout=drop_rate)
        self.layernorm1 = nn.LayerNorm(drug_channels * 3)

        self.lstm2 = nn.LSTM(input_size=drug_channels * 3, hidden_size=drug_channels * 2, batch_first=True, dropout=drop_rate)
        self.layernorm2 = nn.LayerNorm(drug_channels * 2)

        self.lstm3 = nn.LSTM(input_size=drug_channels * 2, hidden_size=512, batch_first=True, dropout=drop_rate)
        self.layernorm3 = nn.LayerNorm(512)

        self.lstm4 = nn.LSTM(input_size=512, hidden_size=128, batch_first=True, dropout=drop_rate)
        self.layernorm4 = nn.LayerNorm(128)

        self.dropout = nn.Dropout(drop_rate)
        self.gelu = nn.GELU()

        # Projection layers for residual connections
        self.proj1_to_2 = nn.Linear(drug_channels * num_modality, drug_channels * 2)
        self.proj2_to_3 = nn.Linear(drug_channels * 2, 128)

        self.output = nn.Linear(128, output_dim)

    def forward(self, x):
        x1, _ = self.lstm1(x)
        x1 = self.layernorm1(x1)
        x1 = self.gelu(x1)
        x1 = self.dropout(x1)

        # LSTM layer 2
        x2, _ = self.lstm2(x1)
        x_projected = self.proj1_to_2(x)
        x2 = self.layernorm2(x2 + x_projected)
        x2 = self.gelu(x2)
        x2 = self.dropout(x2)

        # LSTM layer 3
        x3, _ = self.lstm3(x2)
        x3 = self.layernorm3(x3)
        x3 = self.gelu(x3)
        x3 = self.dropout(x3)

        # LSTM layer 4
        x4, _ = self.lstm4(x3)
        x_projected = self.proj2_to_3(x2)
        x4 = self.layernorm4(x4 + x_projected)
        x4 = self.gelu(x4)
        x4 = self.dropout(x4)

        # Output layer
        x_out = self.output(x4).squeeze(-1)
        return x_out

class MMIDDI(nn.Module):
    def __init__(self, drug_channels, num_heads, num_layers, drop_rate, num_modalities, output_dim):
        super(MMIDDI, self).__init__()

        self.drug_fusion = MMIMFM(model_dim=drug_channels, num_heads=num_heads, num_layers=num_layers, drop_rate=drop_rate)
        self.classifier = Predictor(drug_channels, drop_rate, num_modality=num_modalities, output_dim=output_dim)

    def forward(self, *modalities):
        drug_features1, drug_features2, attention_maps = self.drug_fusion(*modalities)
        out1 = self.classifier(drug_features1)
        out2 = self.classifier(drug_features2)
        out = (out1 + out2) / 2
        return out, attention_maps