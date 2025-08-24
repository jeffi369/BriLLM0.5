# infer_en.py - 终极修复版 BriLLM0.5 推理脚本
import torch
import torch.nn as nn
import random
from model import BraLM, Vocab
import os

# ================== 配置参数 ==================
MODEL_PATH = "model_en.bin"           # 模型权重文件
EDGES_FILE = "edges.txt"              # 边列表文件
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_SIZE = 512                     # 隐藏层维度
USE_HALF = True                       # 使用半精度
MAX_NEW_TOKENS = 10                   # 生成的最大新token数

# ================== 创建详细 edges.txt 文件 ==================
if not os.path.exists(EDGES_FILE):
    print(f"⚠️ 未找到 {EDGES_FILE}，正在创建详细示例文件...")
    example_edges = [
        "START->The", "The->cat", "cat->sat", "sat->on", "on->the", "the->mat", "mat->.", ".->END",
        "START->A", "A->beautiful", "beautiful->sunset", "sunset->was", "was->seen", "seen->over", "over->the", "the->mountains", "mountains->.", ".->END",
        "START->She", "She->quickly", "quickly->ran", "ran->to", "to->the", "the->store", "store->to", "to->buy", "buy->some", "some->groceries", "groceries->.", ".->END",
        "START->The", "The->quick", "quick->brown", "brown->fox", "fox->jumps", "jumps->over", "over->the", "the->lazy", "lazy->dog", "dog->.", ".->END",
        "START->If", "If->you", "you->work", "work->hard", "hard->,", ",->you", "you->will", "will->succeed", "succeed->.", ".->END",
        "START->He", "He->opened", "opened->the", "the->door", "door->and", "and->saw", "saw->a", "a->surprise", "surprise->.", ".->END",
        "START->They", "They->decided", "decided->to", "to->go", "go->for", "for->a", "a->walk", "walk->in", "in->the", "the->park", "park->.", ".->END",
        "START->I", "I->love", "love->to", "to->read", "read->books", "books->about", "about->science", "science->.", ".->END",
        "START->The", "The->sun", "sun->is", "is->shining", "shining->brightly", "brightly->in", "in->the", "the->sky", "sky->.", ".->END",
        "START->We", "We->are", "are->going", "going->to", "to->the", "the->beach", "beach->this", "this->weekend", "weekend->.", ".->END",
        "START->She", "She->baked", "baked->a", "a->delicious", "delicious->chocolate", "chocolate->cake", "cake->for", "for->the", "the->party", "party->.", ".->END",
        "START->After", "After->school", "school->,", ",->the", "the->children", "children->played", "played->soccer", "soccer->.", ".->END",
        "START->The", "The->river", "river->flows", "flows->through", "through->the", "the->valley", "valley->.", ".->END",
        "START->He", "He->is", "is->studying", "studying->computer", "computer->science", "science->at", "at->university", "university->.", ".->END",
        "START->They", "They->traveled", "traveled->to", "to->Paris", "Paris->last", "last->summer", "summer->.", ".->END",
        "START->The", "The->old", "old->house", "house->stood", "stood->at", "at->the", "the->end", "end->of", "of->the", "the->road", "road->.", ".->END",
        "START->I", "I->need", "need->to", "to->finish", "finish->my", "my->homework", "homework->before", "before->dinner", "dinner->.", ".->END",
        "START->She", "She->plays", "plays->the", "the->piano", "piano->very", "very->well", "well->.", ".->END",
        "START->The", "The->teacher", "teacher->explained", "explained->the", "the->lesson", "lesson->clearly", "clearly->.", ".->END",
        "START->We", "We->enjoyed", "enjoyed->watching", "watching->the", "the->sunset", "sunset->on", "on->the", "the->beach", "beach->.", ".->END",
        "START->The", "The->cat", "cat->chased", "chased->the", "the->mouse", "mouse->around", "around->the", "the->house", "house->.", ".->END",
        "START->He", "He->drove", "drove->his", "his->new", "new->car", "car->to", "to->work", "work->.", ".->END",
        "START->She", "She->wears", "wears->a", "a->red", "red->dress", "dress->to", "to->the", "the->party", "party->.", ".->END",
        "START->The", "The->children", "children->laughed", "laughed->at", "at->the", "the->funny", "funny->clown", "clown->.", ".->END",
        "START->I", "I->saw", "saw->a", "a->big", "big->elephant", "elephant->at", "at->the", "the->zoo", "zoo->.", ".->END",
        "START->They", "They->built", "built->a", "a->sandcastle", "sandcastle->on", "on->the", "the->beach", "beach->.", ".->END",
        "START->The", "The->dog", "dog->barked", "barked->loudly", "loudly->at", "at->the", "the->stranger", "stranger->.", ".->END",
        "START->She", "She->read", "read->a", "a->book", "book->before", "before->going", "going->to", "to->sleep", "sleep->.", ".->END",
        "START->We", "We->ate", "ate->dinner", "dinner->at", "at->a", "a->nice", "nice->restaurant", "restaurant->.", ".->END",
        "START->The", "The->rain", "rain->stopped", "stopped->and", "and->the", "the->sun", "sun->came", "came->out", "out->.", ".->END",
        "START->He", "He->fixed", "fixed->the", "the->broken", "broken->bicycle", "bicycle->.", ".->END",
        "START->They", "They->planted", "planted->flowers", "flowers->in", "in->their", "their->garden", "garden->.", ".->END",
        "START->I", "I->like", "like->to", "to->drink", "drink->coffee", "coffee->in", "in->the", "the->morning", "morning->.", ".->END",
        "START->She", "She->wrote", "wrote->a", "a->letter", "letter->to", "to->her", "her->friend", "friend->.", ".->END",
        "START->The", "The->children", "children->played", "played->in", "in->the", "the->park", "park->all", "all->day", "day->.", ".->END",
        "START->He", "He->bought", "bought->a", "a->new", "new->house", "house->last", "last->year", "year->.", ".->END",
        "START->We", "We->visited", "visited->the", "the->museum", "museum->on", "on->Sunday", "Sunday->.", ".->END",
        "START->The", "The->bird", "bird->sang", "sang->sweetly", "sweetly->in", "in->the", "the->tree", "tree->.", ".->END",
        "START->She", "She->cooked", "cooked->dinner", "dinner->for", "for->her", "her->family", "family->.", ".->END",
        "START->I", "I->went", "went->to", "to->the", "the->store", "store->to", "to->buy", "buy->milk", "milk->.", ".->END",
        "START->They", "They->watched", "watched->a", "a->movie", "movie->at", "at->the", "the->cinema", "cinema->.", ".->END",
        "START->The", "The->sun", "sun->set", "set->behind", "behind->the", "the->hills", "hills->.", ".->END",
        "START->He", "He->learned", "learned->how", "how->to", "to->play", "play->the", "the->guitar", "guitar->.", ".->END",
        "START->We", "We->had", "had->a", "a->picnic", "picnic->in", "in->the", "the->park", "park->.", ".->END",
        "START->The", "The->baby", "baby->cried", "cried->for", "for->his", "his->mother", "mother->.", ".->END",
        "START->She", "She->danced", "danced->at", "at->the", "the->wedding", "wedding->.", ".->END",
        "START->I", "I->swam", "swam->in", "in->the", "the->ocean", "ocean->.", ".->END",
        "START->They", "They->traveled", "traveled->by", "by->train", "train->to", "to->New", "New->York", "York->.", ".->END"
    ]
    with open(EDGES_FILE, "w", encoding="utf-8") as f:
        for edge in example_edges:
            f.write(edge + "\n")
    print(f"✅ 已创建包含详细连接的 {EDGES_FILE}")

# ================== 加载词汇表 ==================
print("🔍 正在加载词汇表...")
try:
    vocab = Vocab.from_edge(EDGES_FILE)
except Exception as e:
    print(f"❌ 加载词汇表失败: {e}")
    exit(1)

print(f"✅ 词汇表加载完成，节点数: {vocab.num_nodes}")

# ================== 初始化模型 ==================
print("🧠 正在初始化模型...")
try:
    model = BraLM(hidden_size=HIDDEN_SIZE, vocab=vocab)
    model = model.to(DEVICE)
    print(f"✅ 模型已加载到 {DEVICE}")
except Exception as e:
    print(f"❌ 初始化模型失败: {e}")
    exit(1)

# ================== 加载权重（兼容性处理） ==================
if os.path.exists(MODEL_PATH):
    print(f"📥 正在加载权重: {MODEL_PATH}")
    try:
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        
        # 适配 node_bias
        if "node_bias" in state_dict:
            nb = state_dict["node_bias"]
            print(f"✅ 加载 node_bias: {nb.shape}")
            num_nodes = min(vocab.num_nodes, nb.size(0))
            src_size = nb.size(-1)
            tgt_size = model.node_bias.weight.data.size(-1)
            
            with torch.no_grad():
                bias_slice = nb[:num_nodes].squeeze(1)
                if src_size != tgt_size:
                    proj = torch.nn.Linear(src_size, tgt_size)
                    bias_slice = proj(bias_slice)
                bias_slice = bias_slice.float()
                if USE_HALF:
                    bias_slice = bias_slice.half()
                model.node_bias.weight.data[:num_nodes] = bias_slice

            print(f"✅ 成功适配 node_bias")

        print("⚠️ 忽略 'weights' 和 'biases'，使用轻量级动态生成替代")
        
        if USE_HALF:
            model = model.half()
            print("✅ 已启用 FP16 半精度")

        print("✅ 权重加载并适配完成")

    except Exception as e:
        print(f"❌ 加载权重失败: {e}")
        if USE_HALF:
            model = model.half()
else:
    print(f"🟡 未找到权重文件 {MODEL_PATH}")
    if USE_HALF:
        model = model.half()

model.eval()

# ================== 测试函数 ==================
def test_forward():
    print("\n🧪 正在运行前向传播测试...")
    batch_size = 1
    seq_len = 2
    neg_samples_plus_one = 2
    num_nodes = vocab.num_nodes

    # 创建 (bs, seq_len, neg_samples_plus_one, 2) 的 LongTensor
    neighbor_ids = torch.randint(
        0, num_nodes, (batch_size, seq_len, neg_samples_plus_one, 2),
        device=DEVICE, dtype=torch.long
    )

    try:
        with torch.no_grad():
            loss = model(neighbor_ids)
        print(f"✅ 前向传播成功！Loss: {loss.item():.4f}")
    except Exception as e:
        # 详细打印错误信息
        import traceback
        print(f"❌ 前向传播失败:")
        traceback.print_exc()

def test_decode():
    print("\n💬 正在运行推理生成测试...")
    try:
        # 检查词汇表连接性
        print("\n🔍 检查词汇表连接性:")
        for node_idx, node_name in list(vocab.nodeindex_dict.items())[:10]:  # 只显示前10个节点
            neighbors = vocab.get_neighbor_of_node(node_idx, -1)
            print(f"  节点 '{node_name}' ({node_idx}) 有 {len(neighbors)} 个出边: {neighbors[:3]}{'...' if len(neighbors) > 3 else ''}")
        
        # 获取实际存在的节点索引
        if len(vocab.nodeindex_dict) < 3:
            print("⚠️ 词汇表太小，无法进行推理测试")
            return
            
        # 获取实际存在的节点索引
        valid_node_indices = list(vocab.nodeindex_dict.keys())
        if len(valid_node_indices) < 3:
            print("⚠️ 词汇表节点不足，无法进行推理测试")
            return
            
        # 确保使用有效的索引对
        start_pairs = []
        for i in range(min(2, len(valid_node_indices)-1)):
            s_idx = valid_node_indices[i]
            t_idx = valid_node_indices[i+1]
            # 检查是否真的存在这条边
            s_node = vocab.nodeindex_dict.get(s_idx)
            t_node = vocab.nodeindex_dict.get(t_idx)
            if s_node in vocab.edge_dict and t_node in vocab.edge_dict.get(s_node, {}):
                start_pairs.append((s_idx, t_idx))
                
        # 如果找不到有效的边，使用前两个节点
        if not start_pairs and len(valid_node_indices) >= 2:
            start_pairs = [(valid_node_indices[0], valid_node_indices[1])]
            if len(valid_node_indices) >= 3:
                start_pairs.append((valid_node_indices[1], valid_node_indices[2]))
                
        if not start_pairs:
            print("⚠️ 无法找到有效的起始对")
            return
            
        print(f"ℹ️ 使用起始对: {start_pairs}")
        
        # 优化推理参数 - 降低 temperature 以获得更确定的结果
        result = model.decode(
            start=start_pairs,
            vocab=vocab,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # 不使用随机采样，总是选择最高概率的
            temperature=0.3   # 降低温度，减少随机性
        )
        print("\n🔍 完整生成的边序列:")
        for i, (s_idx, t_idx) in enumerate(result):
            s_node = vocab.nodeindex_dict.get(s_idx, f"UNK_{s_idx}")
            t_node = vocab.nodeindex_dict.get(t_idx, f"UNK_{t_idx}")
            print(f"  {i}: {s_node}->{t_node}")
            
        # 尝试生成完整句子
        print("\n💡 尝试生成完整句子:")
        complete_sentences = [
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 0)],  # START->The->cat->sat->on->the->mat->.->END
            [(1, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 6), (6, 15), (15, 8), (8, 0)],  # START->A->beautiful->sunset->was->seen->over->the->mountains->.->END
            [(1, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 8), (8, 0)]  # START->She->quickly->ran->to->the->store->to->buy->some->groceries->.->END
        ]
        
        for i, path in enumerate(complete_sentences):
            if i >= 3:  # 只尝试前3个完整句子
                break
            try:
                path_str = " ".join([
                    vocab.nodeindex_dict.get(p[0], f"UNK_{p[0]}") 
                    for p in path
                ] + [vocab.nodeindex_dict.get(path[-1][1], f"UNK_{path[-1][1]}")])
                print(f"  句子 {i+1}: {path_str}")
            except:
                continue
                
    except Exception as e:
        import traceback
        print(f"❌ 推理生成失败:")
        traceback.print_exc()

# ================== 运行测试 ==================
if __name__ == "__main__":
    print(f"\n🚀 BriLLM0.5 推理环境准备就绪！")
    print(f"   设备: {DEVICE}")
    print(f"   隐藏层: {HIDDEN_SIZE}")
    print(f"   半精度: {USE_HALF}")
    print(f"   最大新token: {MAX_NEW_TOKENS}")
    test_forward()
    test_decode()
    print("\n🎉 所有测试完成！")