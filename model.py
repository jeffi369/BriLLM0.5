# model.py - 终极修复版 BriLLM0.5 (模拟大脑推理架构)
import torch
import torch.nn as nn
import random

class BraLM(nn.Module):
    def __init__(self, hidden_size=512, embed_dim=256, rank=8, use_ds=False, vocab=None):
        """
        模拟大脑推理架构的轻量版 BriLLM0.5
        专为解决图结构语言模型的维度问题设计
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.rank = rank
        self.activation = nn.GELU()
        self.positions = nn.Parameter(torch.ones(1, 512, 1))
        self.device = None
        self.use_ds = use_ds
        self.vocab = vocab

        # 核心：用嵌入 + 低秩变换替代巨大参数矩阵
        self.node_embeddings = nn.Embedding(vocab.num_nodes, embed_dim)
        self.W_left = nn.Linear(embed_dim, hidden_size * rank)
        self.W_right = nn.Linear(embed_dim, hidden_size * rank)
        self.bias_base = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.node_bias = nn.Embedding(vocab.num_nodes, hidden_size)

        # 参数初始化
        nn.init.normal_(self.node_embeddings.weight, std=0.02)
        nn.init.normal_(self.bias_base, std=0.02)
        nn.init.normal_(self.node_bias.weight, std=0.02)

    def prepare_network(self, vocab):
        self.vocab = vocab
        print(f"🧠 模型已准备，节点数: {vocab.num_nodes}, 隐藏层: {self.hidden_size}")

    def to_device(self, device):
        self.device = device
        self = self.to(device)

    def get_positional_encoding(self, seq_len, d_model):
        """生成正弦位置编码"""
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = 10000.0 ** (torch.arange(0, d_model, 2) / d_model)
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(self.device)
        if self.node_embeddings.weight.dtype == torch.float16:
            pe = pe.half()
        return pe

    def get_initial_tensor(self, batch_size, src_node_idx, pe, pos=0):
        """初始化能量张量"""
        node_emb = self.node_embeddings(src_node_idx[:, 0])
        energy_tensor = self.activation(self.node_bias(src_node_idx[:, 0]))
        energy_tensor = energy_tensor.unsqueeze(1)
        pe = pe[:, pos] if pe.dim() == 3 else pe
        return self.activation(energy_tensor + pe[:batch_size])

    def forward(self, neighbor_ids):
        """
        前向传播 - 修复了维度问题
        :param neighbor_ids: (bs, seq_len, 1+k, 2)  # 2 表示 (src_idx, tgt_idx)
        """
        batch_size, seq_len = neighbor_ids.shape[:2]
        device = neighbor_ids.device
        self.to_device(device)

        pe = self.get_positional_encoding(512, self.hidden_size)
        energy_cache = []
        loss = 0.0

        for i in range(seq_len):
            d = neighbor_ids[:, i]
            src_idx = d[..., 0]
            tgt_idx = d[..., 1]
            num_neg_samples = tgt_idx.size(1)  # 获取负样本数 (1+k)

            if i == 0:
                energy_tensor = self.get_initial_tensor(batch_size, src_idx, pe, 0)
            else:
                cache_tensor = torch.stack(energy_cache, dim=1)
                weights = self.positions[:, :i].softmax(dim=1)
                energy_tensor = (cache_tensor * weights).sum(dim=1, keepdim=True)
                
                # 确保 energy_tensor 是 3D [bs, 1, D]
                if energy_tensor.dim() > 3:
                    energy_tensor = energy_tensor.squeeze(2)

            # 确保 energy_tensor 是 3D [bs, 1, D]
            if energy_tensor.dim() == 2:
                energy_tensor = energy_tensor.unsqueeze(1)
            elif energy_tensor.dim() == 1:
                energy_tensor = energy_tensor.unsqueeze(0).unsqueeze(0)
            # 如果已经是 4D，去掉多余的维度
            elif energy_tensor.dim() > 3:
                energy_tensor = energy_tensor.squeeze(2)

            # 获取节点嵌入
            src_emb = self.node_embeddings(src_idx)
            tgt_emb = self.node_embeddings(tgt_idx)

            # 生成低秩权重矩阵 W ∈ R^(bs*(1+k), D, D)
            U = self.W_left(src_emb).view(-1, self.rank, self.hidden_size)
            V = self.W_right(tgt_emb).view(-1, self.rank, self.hidden_size)
            W = torch.bmm(U.transpose(1, 2), V) / (self.rank ** 0.5)

            # 生成偏置 b ∈ R^(bs, 1+k, 1, D)
            bias = self.bias_base + self.node_bias(tgt_idx).unsqueeze(2)

            # 关键修复：正确扩展 energy_tensor
            # (bs, 1, D) -> (bs, 1+k, 1, D) -> (bs*(1+k), 1, D)
            expand_energy = energy_tensor.unsqueeze(1).expand(-1, num_neg_samples, -1, -1)
            expand_energy = expand_energy.reshape(-1, 1, self.hidden_size)

            nxt_energy = torch.bmm(expand_energy, W) + bias.reshape(-1, 1, self.hidden_size)
            nxt_energy = self.activation(nxt_energy + pe[:, i+1].unsqueeze(0))

            output_tensor = nxt_energy.reshape(batch_size, -1, 1, self.hidden_size)

            if i == 0:
                energy_cache = [output_tensor[:, 0]]
            else:
                energy_cache.append(output_tensor[:, 0])

            energy = output_tensor.norm(p=2, dim=(-2, -1))
            label = torch.zeros(batch_size, dtype=torch.long, device=device)
            loss += nn.CrossEntropyLoss()(energy, label)

        return loss / seq_len

    def decode(self, start, vocab, max_new_tokens=16, do_sample=False, temperature=1.0):
        """
        推理生成 - 修复了起始点问题
        :param start: 起始边列表，如 [(0,1), (1,2)]
        :return: 生成的边序列
        """
        ret = []
        device = next(self.parameters()).device
        self.to_device(device)
        pe = self.get_positional_encoding(512, self.hidden_size).squeeze(0)
        energy_cache = []

        for i, (s, t) in enumerate(start):
            if i == 0:
                src_idx = torch.tensor([[s]], device=device)
                energy_tensor = self.get_initial_tensor(1, src_idx, pe.unsqueeze(0), 0).squeeze(0)
            else:
                cache_tensor = torch.stack(energy_cache, dim=1)
                weights = self.positions[:, :i].softmax(dim=1)
                energy_tensor = (cache_tensor * weights).sum(dim=1, keepdim=True)
                
                # 确保 energy_tensor 是 2D [1, D]
                if energy_tensor.dim() > 2:
                    energy_tensor = energy_tensor.squeeze(0).squeeze(0)
                elif energy_tensor.dim() == 2:
                    energy_tensor = energy_tensor.squeeze(0)

            # 获取嵌入
            s_emb = self.node_embeddings(torch.tensor([s], device=device))
            t_emb = self.node_embeddings(torch.tensor([t], device=device))
            U = self.W_left(s_emb).view(self.rank, self.hidden_size)
            V = self.W_right(t_emb).view(self.rank, self.hidden_size)
            W = torch.mm(U.transpose(0, 1), V) / (self.rank ** 0.5)
            b = self.bias_base + self.node_bias(torch.tensor([t], device=device)).unsqueeze(1)

            if energy_tensor.dim() == 1:
                energy_tensor = energy_tensor.unsqueeze(0)

            nxt_energy = torch.mm(energy_tensor, W) + b.squeeze(0)
            energy_tensor = self.activation(nxt_energy + pe[i+1].unsqueeze(0)).squeeze(0)

            if i == 0:
                energy_cache = [energy_tensor.unsqueeze(0).unsqueeze(0)]
            else:
                energy_cache.append(energy_tensor.unsqueeze(0).unsqueeze(0))

            ret.append((s, t))

        x = t
        for i in range(max_new_tokens):
            candidates = vocab.get_neighbor_of_node(x, -1)
            if not candidates:
                break

            # 处理候选节点
            candidate_indices = []
            for c in candidates:
                if "->" in c:
                    _, t_str = c.split("->", 1)
                    t_idx = vocab.node_dict.get(t_str)
                    if t_idx is not None:
                        candidate_indices.append(t_idx)
            
            if not candidate_indices:
                break
                
            tgt_indices = torch.tensor(candidate_indices, device=device)
            x_emb = self.node_embeddings(torch.tensor([x], device=device))
            t_embs = self.node_embeddings(tgt_indices)

            U = self.W_left(x_emb).view(self.rank, self.hidden_size)
            V = self.W_right(t_embs).view(-1, self.rank, self.hidden_size)
            W = torch.bmm(U.unsqueeze(0).expand(len(t_embs), -1, -1).transpose(1, 2), V) / (self.rank ** 0.5)
            b = self.bias_base + self.node_bias(tgt_indices).unsqueeze(1)

            curr_i = len(start) + i
            current_energy = (torch.cat(energy_cache, dim=1) * self.positions[:, :curr_i].softmax(1)).sum(1)
            
            # 确保 current_energy 是 2D [1, D]
            if current_energy.dim() > 2:
                current_energy = current_energy.squeeze(0)
            elif current_energy.dim() == 1:
                current_energy = current_energy.unsqueeze(0)
                
            expand_energy = current_energy.unsqueeze(1).repeat(len(W), 1, 1)

            if expand_energy.dim() == 2:
                expand_energy = expand_energy.unsqueeze(1)

            nxt_energy = torch.bmm(expand_energy, W) + b
            output_tensor = self.activation(nxt_energy + pe[curr_i+1].unsqueeze(0).unsqueeze(0))
            energy = output_tensor.norm(p=2, dim=(1, 2))
            probs = torch.softmax(energy / (temperature + 1e-8), dim=-1)

            if do_sample:
                index = torch.multinomial(probs, 1).item()
            else:
                index = probs.argmax().item()

            # 从 candidates 中获取下一个节点
            if "->" in candidates[index]:
                _, next_node_str = candidates[index].split("->", 1)
                y = vocab.node_dict.get(next_node_str, x)
            else:
                y = x

            ret.append((x, y))
            energy_cache.append(output_tensor[index].unsqueeze(0))
            x = y

        return ret


class Vocab:
    def __init__(self, node_dict, nodeindex_dict, edge_dict, edge_decode_dict):
        self.node_dict = node_dict
        self.nodeindex_dict = nodeindex_dict
        self.edge_dict = edge_dict
        self.edge_decode_dict = edge_decode_dict

        # 处理 None 值
        if self.nodeindex_dict is None:
            self.nodeindex_dict = {}
            node_id = 0
            for s in self.edge_dict:
                if s == "" or s not in self.nodeindex_dict.values():
                    self.nodeindex_dict[node_id] = s
                    node_id += 1
        if self.node_dict is None:
            self.node_dict = {v: k for k, v in self.nodeindex_dict.items()}

        self.num_nodes = len(self.nodeindex_dict)

    @classmethod
    def from_edge(cls, filename):
        edge_dict = {"": {"": (0, 0)}}
        edge_decode_dict = {(0, 0): ""}

        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue
                if '->' not in line:
                    print(f"⚠️ 无效行 {line_num}: '{line}' (缺少 '->')")
                    continue

                try:
                    s, t = line.split('->', 1)
                    s, t = s.strip(), t.strip()
                    if not s or not t:
                        print(f"⚠️ 节点为空 {line_num}: '{s}' -> '{t}'")
                        continue

                    if s not in edge_dict:
                        index_s = len(edge_dict)
                        edge_dict[s] = {}
                    else:
                        index_s = next(iter(edge_dict[s].values()))[0]

                    if t not in edge_dict[s]:
                        index_t = len(edge_dict[s])
                    else:
                        index_t = edge_dict[s][t][1]

                    edge_dict[s][t] = (index_s, index_t)
                    edge_decode_dict[(index_s, index_t)] = f"{s}->{t}"

                except Exception as e:
                    print(f"解析行 {line_num} 失败: {line} | 错误: {e}")
                    continue

        # 构建 node_dict 和 nodeindex_dict
        node_dict = {}
        nodeindex_dict = {}
        seen_nodes = set()

        for s in edge_dict:
            if s == "":
                continue
            seen_nodes.add(s)
            for t in edge_dict[s]:
                idx_s = edge_dict[s][t][0]
                node_dict[s] = idx_s
                nodeindex_dict[idx_s] = s
                break

        if "" not in node_dict:
            node_dict[""] = 0
            nodeindex_dict[0] = ""

        print(f"✅ 成功加载 {len(seen_nodes)} 个节点，{len(edge_decode_dict)} 条边")
        return cls(node_dict, nodeindex_dict, edge_dict, edge_decode_dict)

    def get_neighbor_of_node(self, key, k):
        s = self.nodeindex_dict.get(key)
        if not s or s not in self.edge_dict:
            return []
        neighbors = [f"{s}->{t}" for t in self.edge_dict[s].keys() if t != s]
        random.shuffle(neighbors)
        return neighbors[:k] if k != -1 else neighbors

    def decode(self, x):
        return self.edge_decode_dict.get(x, f"UNK_{x}")

    def __call__(self, x):
        if isinstance(x, list):
            return [self.__call__(_) for _ in x]
        else:
            return self.fetch(x)

    def fetch(self, x):
        s, t = x.split("->")
        return self.edge_dict.get(s, {}).get(t, (0, 0))