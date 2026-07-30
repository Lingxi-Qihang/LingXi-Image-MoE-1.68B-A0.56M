import logging
import os
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
import torch.nn.functional as F
from .modules import Attention, modulate, TimestepEmbedder, FinalLayer, MoeMLP, Mlp, precompute_2d_rope, precompute_rope_1d


#################################################################################
#                                ProMoE Layer                                  #
#################################################################################
class AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class SparseMoeBlock(nn.Module):
    _moe_logger = None
    def __init__(
            self,
            num_routed_experts,
            hidden_size,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            top_k=2,
            load_balance_loss_coef=0,
            norm_topk_prob=False,
            seq_aux=False,
            use_shared_expert=True,
            use_uncond_expert=True,
            router_weight_mode="softmax",
            routing_contrastive_lam=0,
            use_top_k_for_routing_contrastive=False,
            routing_contrastive_temperature=0.1,
            layer_idx=None,
            **kwargs,
    ):
        super().__init__()
        if use_uncond_expert:
            self.num_experts = num_routed_experts + 1
        else:
            self.num_experts = num_routed_experts
        self.num_routed_experts = num_routed_experts
        self.seq_aux = seq_aux
        self.hidden_size = hidden_size
        self.top_k = top_k

        self.cluster_centers = nn.Parameter(torch.randn(num_routed_experts, hidden_size))

        self.alpha = load_balance_loss_coef
        self.use_shared_expert = use_shared_expert
        self.use_uncond_expert = use_uncond_expert
        self.router_weight_mode = router_weight_mode

        self.routing_contrastive_lam = routing_contrastive_lam
        self.use_top_k_for_routing_contrastive = use_top_k_for_routing_contrastive
        self.routing_contrastive_temperature = routing_contrastive_temperature

        self.experts = nn.ModuleList(
            [MoeMLP(hidden_size=hidden_size, intermediate_size=moe_intermediate_size)
             for _ in range(self.num_experts)]
        )

        if use_shared_expert:
            self.shared_expert = MoeMLP(
                hidden_size=hidden_size,
                intermediate_size=shared_expert_intermediate_size
            )

        self._init_weights()
        # 初始化类级别的 MoE 日志（仅执行一次）
        if SparseMoeBlock._moe_logger is None:
            log_dir = 'logs'
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'moe_routing.log')

            logger = logging.getLogger('MoE_Routing')
            logger.setLevel(logging.INFO)
            logger.propagate = False
            if logger.handlers:
                logger.handlers.clear()
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
            logger.addHandler(file_handler)

            SparseMoeBlock._moe_logger = logger

        # 每个实例都引用同一个 logger
        self.moe_logger = SparseMoeBlock._moe_logger

        self.layer_idx=layer_idx


    def compute_router(self, hidden_states, labels):
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        flat_input = hidden_states.view(-1, self.hidden_size)

        # 处理 labels：可能是 [B] 的文本特征标记，也可能是 [B, L] 的 token 级标记
        if labels.dim() == 1:
            # [B] → [B, L]
            flat_labels = labels.view(batch_size, 1).expand(-1, seq_len).reshape(-1)
        elif labels.dim() == 2 and labels.shape == (batch_size, seq_len):
            flat_labels = labels.reshape(-1)
        else:
            # fallback: 全部视为有条件
            flat_labels = torch.zeros(batch_size * seq_len, dtype=torch.long, device=device)

        if self.use_uncond_expert:
            # 0 = 有条件（正常文本），1 = 无条件（空文本/CFG）
            uncond_mask = (flat_labels == 1)
            cond_mask = ~uncond_mask
        else:
            uncond_mask = None
            cond_mask = torch.ones_like(flat_labels, dtype=torch.bool)

        router_weights = torch.zeros(batch_size * seq_len, self.top_k, device=device)
        expert_indices = torch.zeros(batch_size * seq_len, self.top_k, device=device, dtype=torch.long)

        if uncond_mask is not None and uncond_mask.any():
            uncond_positions = torch.where(uncond_mask)[0]
            router_weights[uncond_positions, 0] = 1.0
            expert_indices[uncond_positions] = self.num_experts - 1

        if cond_mask.any():
            cond_positions = torch.where(cond_mask)[0]
            cond_input = flat_input[cond_positions]

            input_norm = F.normalize(cond_input, p=2, dim=1)
            cluster_norm = F.normalize(self.cluster_centers, p=2, dim=1)
            cos_sim = input_norm @ cluster_norm.T

            if self.router_weight_mode == "softmax":
                cond_weights = F.softmax(cos_sim, dim=1)
            elif self.router_weight_mode == "sigmoid":
                sigmoid_scale = 1.0
                cond_weights = torch.sigmoid(cos_sim * sigmoid_scale)
            elif self.router_weight_mode == "identity":
                cond_weights = cos_sim
            else:
                raise ValueError(f"Unsupported router_weight_mode: {self.router_weight_mode}")

            topk_scores, topk_idx = torch.topk(cond_weights, k=self.top_k, dim=1)
            router_weights[cond_positions] = topk_scores.to(router_weights.dtype)
            expert_indices[cond_positions] = topk_idx

        router_weights = router_weights.view(batch_size, seq_len, self.top_k)
        expert_indices = expert_indices.view(batch_size, seq_len, self.top_k)

        load_balance_loss = None  # ProMoE 不使用 load balancing loss
        if self.training and self.alpha > 0.0:
            # 保留原逻辑但通常不会触发
            cond_batch_size = cond_mask.sum().item()
            if cond_batch_size > 0 and self.router_weight_mode != "softmax":
                scores_for_aux = F.softmax(cond_weights, dim=1)
                mask_ce = F.one_hot(topk_idx.view(-1), num_classes=self.num_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.num_routed_experts
                load_balance_loss = (Pi * fi).sum() * self.alpha

        return router_weights, expert_indices, load_balance_loss

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor,global_step: int = None):
        router_weights, expert_indices, load_balance_loss = self.compute_router(hidden_states, labels)
        batch_size, seq_len, hidden_dim = hidden_states.shape

        flat_input = hidden_states.view(-1, hidden_dim)
        flat_weights = router_weights.view(-1, self.top_k)
        flat_indices = expert_indices.view(-1, self.top_k)
        total_tokens = batch_size * seq_len

        final_output = torch.zeros(total_tokens, hidden_dim, device=hidden_states.device)

        for expert_id in range(self.num_experts):
            expert_mask = (flat_indices == expert_id).any(dim=1)
            token_ids = torch.where(expert_mask)[0]
            if token_ids.numel() > 0:
                expert_input = flat_input[token_ids]
                expert_weight_mask = (flat_indices[token_ids] == expert_id)
                expert_weights = flat_weights[token_ids] * expert_weight_mask.float()
                combined_weights = expert_weights.sum(dim=1)
                expert_output = self.experts[expert_id](expert_input)
                weighted_output = expert_output * combined_weights.unsqueeze(1)
                final_output.index_add_(0, token_ids, weighted_output)
            else:
                dummy_input = torch.zeros(1, hidden_dim, device=hidden_states.device)
                dummy_output = self.experts[expert_id](dummy_input).float()
                final_output[0] += dummy_output[0] * 0

        final_output = final_output.view(batch_size, seq_len, hidden_dim)

        if self.use_shared_expert:
            shared_output = self.shared_expert(hidden_states)
            final_output += shared_output

        loss = load_balance_loss

        if self.training and self.routing_contrastive_lam > 0:
            flat_labels = labels.view(batch_size, 1).expand(-1, seq_len).reshape(
                -1) if labels.dim() == 1 else labels.reshape(-1)
            if self.use_uncond_expert:
                uncond_mask = (flat_labels == 1)
                cond_mask = ~uncond_mask
            else:
                cond_mask = torch.ones(batch_size * seq_len, dtype=torch.bool, device=hidden_states.device)

            cond_token_embeddings = flat_input[cond_mask]

            if cond_mask.sum() > 0:
                if self.use_top_k_for_routing_contrastive:
                    topk_expert_indices = expert_indices.view(batch_size * seq_len, self.top_k)[cond_mask]
                    cond_cluster_assignments = topk_expert_indices
                else:
                    top1_expert_indices = expert_indices.view(batch_size * seq_len, self.top_k)[:, 0]
                    cond_cluster_assignments = top1_expert_indices[cond_mask]

                routing_contrastive_loss = self.compute_routing_contrastive_loss(
                    cond_token_embeddings,
                    cond_cluster_assignments,
                    use_top_k=self.use_top_k_for_routing_contrastive
                )
                routing_contrastive_loss = routing_contrastive_loss * self.routing_contrastive_lam
                loss = routing_contrastive_loss if loss is None else loss + routing_contrastive_loss

        # ---------- 记录路由日志 ----------
        if self.training and global_step is not None and global_step % 100 == 0:
            with torch.no_grad():
                # 基于 top-1 统计每个专家的token数
                top1_indices = expert_indices[:, :, 0].reshape(-1)  # [B*L]
                total = top1_indices.numel()
                expert_counts = torch.bincount(top1_indices, minlength=self.num_experts)
                utilization = {str(i): round(cnt.item() / total, 4) for i, cnt in enumerate(expert_counts)}
                self.moe_logger.info(f"Step {global_step}, Layer {self.layer_idx}: {utilization}")

        return final_output, loss

    def compute_routing_contrastive_loss(self, token_embeddings, cluster_assignments, use_top_k=False):
        cluster_centers = self.cluster_centers
        num_clusters = cluster_centers.size(0)
        device = cluster_centers.device

        cluster_means = []
        valid_clusters = []

        for cluster_id in range(num_clusters):
            if use_top_k:
                mask = (cluster_assignments == cluster_id).any(dim=1)
            else:
                mask = (cluster_assignments == cluster_id)
            if mask.sum() > 0:
                cluster_mean = token_embeddings[mask].mean(dim=0, keepdim=True)
                cluster_means.append(cluster_mean)
                valid_clusters.append(cluster_id)

        if len(valid_clusters) < 2:
            return torch.tensor(0.0, device=device)

        cluster_means = torch.cat(cluster_means, dim=0)
        valid_centers = cluster_centers[valid_clusters]
        centers_norm = F.normalize(valid_centers, p=2, dim=1)
        means_norm = F.normalize(cluster_means, p=2, dim=1)
        sim_matrix = centers_norm @ means_norm.T

        temperature = self.routing_contrastive_temperature
        labels = torch.arange(sim_matrix.size(0), device=device)
        logits = sim_matrix / temperature

        return F.cross_entropy(logits, labels)

    def _init_weights(self):
        nn.init.normal_(self.cluster_centers, mean=0.0, std=0.02)


#################################################################################
#                                 Core ProMoE Model                            #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, head_dim=None, mlp_ratio=4.0,
                 use_swiglu=False, MoE_config=None, use_moe=False,layer_idx=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, head_dim=head_dim, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.use_moe = use_moe
        if use_moe:
            self.mlp = SparseMoeBlock(hidden_size=hidden_size, layer_idx=layer_idx,**MoE_config)
        else:
            if use_swiglu:
                self.mlp = MoeMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim)
            else:
                self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
                               act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, label, global_step=None, rope_cos=None, rope_sin=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        # 1. 自注意力
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope_cos=rope_cos, rope_sin=rope_sin,
        )

        # 3. MoE / FFN
        if self.use_moe:
            x_mlp, aux_loss = self.mlp(
                modulate(self.norm2(x), shift_mlp, scale_mlp), label,global_step=global_step
            )
            if aux_loss is not None:
                x_mlp = AddAuxiliaryLoss.apply(x_mlp, aux_loss)
            x = x + gate_mlp.unsqueeze(1) * x_mlp
        else:
            x = x + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm2(x), shift_mlp, scale_mlp)
            )
        return x


class DiT(nn.Module):
    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            qk_norm=False,
            class_dropout_prob=0.1,
            num_classes=1000,
            learn_sigma=True,
            use_swiglu=False,
            MoE_config=None,
            head_dim=None,
            # 🆕 文生图参数
            text_embed_dim=768,  # T5-base d_model
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.MoE_config = MoE_config
        use_moe_flag = [True] * depth
        if self.MoE_config and self.MoE_config.get('interleave', False):
            use_moe_flag = [i % 2 == 1 for i in range(depth)]

        #use_moe_flag = [False] * depth
        use_moe_flag = [True] * depth

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        # 覆盖 forward 以支持动态分辨率 (去掉尺寸断言)
        def _flexible_pe_forward(x):
            B, C, H, W = x.shape
            x = self.x_embedder.proj(x)
            if self.x_embedder.flatten:
                x = x.flatten(2).transpose(1, 2)
            if self.x_embedder.norm:
                x = self.x_embedder.norm(x)
            return x
        self.x_embedder.forward = _flexible_pe_forward

        # head_dim for RoPE
        if head_dim is not None:
            self._rope_head_dim = head_dim
        else:
            self._rope_head_dim = hidden_size // num_heads

        self.t_embedder = TimestepEmbedder(hidden_size)

        # 🆕 替换 LabelEmbedder 为文本投影层
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # 可学习的空文本嵌入（用于 CFG）
        self.null_text_embed = nn.Parameter(torch.randn(1, text_embed_dim) * 0.02)
        self.class_dropout_prob = class_dropout_prob

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size, num_heads,
                head_dim=head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm,
                use_swiglu=use_swiglu, MoE_config=MoE_config, use_moe=use_moe_flag[i],layer_idx=i
            ) for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.init_MoeMLP = MoE_config.get('init_MoeMLP', False) if MoE_config else False

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 🆕 初始化文本投影层
        for module in self.text_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.init_MoeMLP:
            def init_MoeMLP(module, std=0.006):
                nn.init.normal_(module.gate_proj.weight, std=std)
                nn.init.normal_(module.up_proj.weight, std=std)
                nn.init.normal_(module.down_proj.weight, std=std)

            for block in self.blocks:
                for expert in block.mlp.experts:
                    init_MoeMLP(expert)
            print("init MoE related module with std 0.006 like DeepSeek-MoE")

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, timestep, router_labels=None, text_sequence=None, text_mask=None, source_latent=None,global_step=None,
                **kwargs):
        """
        Forward pass of DiT (序列共生模式，同时支持文生图和图像编辑).

        Args:
            x: (N, C, H, W) 目标潜变量（加噪后的）
            timestep: (N,) 时间步
            router_labels: (N,) MoE 路由标签，0=有条件 1=无条件
            text_sequence: (N, L_txt, text_embed_dim) T5 完整序列
            text_mask: (N, L_txt) 文本注意力掩码，1=有效 0=填充
            source_latent: (N, C, H, W) 源图像潜变量（文生图时为黑图，编辑时为原图）
        Returns:
            (N, C_out, H, W) 去噪后的目标图像
        """
        if len(x.shape) != 4:
            x = x.squeeze(2)

        # 1. 目标图像 token 化 (无位置编码，由 RoPE 替代)
        grid_h, grid_w = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        x = self.x_embedder(x)  # (N, T_img, hidden_size)
        t = self.t_embedder(timestep)  # (N, hidden_size)
        head_dim = self._rope_head_dim

        # 2. 🆕 源图像拼接（如果有 source_latent，拼在目标前面）
        T_src = 0
        if source_latent is not None:
            if len(source_latent.shape) != 4:
                source_latent = source_latent.squeeze(2)
            source_x = self.x_embedder(source_latent[:,:self.in_channels,:,:])  # (N, T_src, hidden_size)
            T_src = source_x.shape[1]
            x = torch.cat([source_x, x], dim=1)  # (N, T_src + T_img, hidden_size)

        # 3. 文本拼接（无 text_pos_embed，由 1D RoPE 替代）
        if text_sequence is not None and text_mask is not None:
            text_seq = self.text_proj(text_sequence)
            L_txt = text_seq.shape[1]
            x = torch.cat([text_seq, x], dim=1)  # (N, L_txt + T_src + T_img, hidden_size)
            context = (text_sequence * text_mask.unsqueeze(-1).float()).sum(dim=1) / \
                      text_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        else:
            # 无条件分支
            text_seq = self.null_text_embed.unsqueeze(0).expand(x.shape[0], 1, -1)
            text_seq = self.text_proj(text_seq)
            L_txt = text_seq.shape[1]
            x = torch.cat([text_seq, x], dim=1)
            context = self.null_text_embed.expand(x.shape[0], -1)

        # 4. 路由标签初始化
        if router_labels is None:
            router_labels = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        # 5. CFG 替换（只影响全局 context）
        if context.dim() == 2 and context.shape[-1] == self.text_proj[0].in_features:
            if self.training and self.class_dropout_prob > 0:
                mask = torch.rand(context.shape[0], device=context.device) < self.class_dropout_prob
                context = torch.where(
                    mask.unsqueeze(-1).expand_as(context),
                    self.null_text_embed.expand(context.shape[0], -1),
                    context,
                )
                router_labels = router_labels.clone()
                router_labels[mask] = 1
            y = self.text_proj(context)
        else:
            y = context

        c = t + y  # AdaLN 调制向量

        # 5.5 计算 RoPE (连续位置编码，天然支持任意分辨率)
        # 文本：1D RoPE (基于序列位置)
        text_pos = torch.arange(L_txt, device=x.device)
        rope_cos_text, rope_sin_text = precompute_rope_1d(text_pos, head_dim)
        # 图像：2D RoPE (归一化坐标到 [0,1]，保证所有分辨率频率范围一致)
        h_lin = torch.linspace(0, 1, grid_h, device=x.device)
        w_lin = torch.linspace(0, 1, grid_w, device=x.device)
        h_pos = h_lin.view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        w_pos = w_lin.view(1, -1).expand(grid_h, grid_w).reshape(-1)
        rope_cos_img, rope_sin_img = precompute_2d_rope(h_pos, w_pos, head_dim)
        # 拼接: text + source + target (source 与 target 共享相同网格)
        if T_src > 0:
            rope_cos = torch.cat([rope_cos_text, rope_cos_img, rope_cos_img], dim=0)
            rope_sin = torch.cat([rope_sin_text, rope_sin_img, rope_sin_img], dim=0)
        else:
            rope_cos = torch.cat([rope_cos_text, rope_cos_img], dim=0)
            rope_sin = torch.cat([rope_sin_text, rope_sin_img], dim=0)

        # 6. Transformer Blocks (传递 RoPE)
        for block in self.blocks:
            x = block(x, c, router_labels, global_step=global_step,
                      rope_cos=rope_cos, rope_sin=rope_sin)

        # 7. 输出：移除文本和源图像，只保留目标图像 token
        x = x[:, (L_txt + T_src):, :]

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x