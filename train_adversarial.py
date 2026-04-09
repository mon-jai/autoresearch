import os
import time
import math
import torch
import torch.nn.functional as F

# 關閉 HF 進度條並開啟 PyTorch 記憶體碎片整理以對齊 DGX GB10
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from models.transformer import GPTConfig
from models.encoder import KGEncoder, KBGANGenerator
from models.decoder import KGDecoder, RealismCritic

# -----------------------------------------------------------------------------
# Configuration parameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 16
MAX_STEPS = 1000
LEARNING_RATE = 3e-4
CRITIC_LR = 1e-4      # Critic learns slower → prevents it from dominating
MAX_GRAD_NORM = 1.0
LABEL_SMOOTH = 0.1    # Real targets = 0.9, fake targets = 0.1 (prevents overconfident Critic)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Structured Synthetic Data (替代純隨機 tensor，讓負採樣有意義)
# -----------------------------------------------------------------------------
class StructuredKGData:
    """
    建立固定的 entity/relation embedding table，模擬有結構的知識圖譜。
    KBGAN 核心洞見：均勻隨機負樣本太容易區分 → 梯度枯竭。
    結構化數據讓 hard negative mining 有語義鄰域可操作。
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device
        # 固定的 entity/relation prototype embeddings (模擬語義空間)
        self.entity_protos = torch.randn(config.num_entities, config.num_entities, device=device) * 0.3
        # 對角線加強：每個 entity 的自身維度有更高值
        self.entity_protos += torch.eye(config.num_entities, device=device) * 1.0
        self.entity_protos = F.normalize(self.entity_protos, dim=-1)

        self.relation_protos = torch.randn(config.num_relations, config.num_relations, device=device) * 0.3
        self.relation_protos += torch.eye(config.num_relations, device=device) * 1.0
        self.relation_protos = F.normalize(self.relation_protos, dim=-1)

    def sample(self, batch_size):
        """採樣結構化的 batch 數據"""
        # 隨機選取 entity/relation indices 作為 positive 三元組
        ent_idx = torch.randint(0, self.config.num_entities, (batch_size,), device=self.device)
        rel_idx = torch.randint(0, self.config.num_relations, (batch_size,), device=self.device)

        # 從 prototype table 查表 + 加入微量噪聲
        gt_entities = self.entity_protos[ent_idx] + torch.randn(batch_size, self.config.num_entities, device=self.device) * 0.05
        gt_relations = self.relation_protos[rel_idx] + torch.randn(batch_size, self.config.num_relations, device=self.device) * 0.05

        # 模擬文字輸入 Token
        real_text_ids = torch.randint(0, self.config.vocab_size, (batch_size, self.config.sequence_len), device=self.device)

        # 模擬知識圖譜特徵矩陣
        gt_kg_features = torch.randn(batch_size, self.config.n_embd, device=self.device)

        return real_text_ids, gt_kg_features, gt_entities, gt_relations

def train():
    print(f"啟動對抗訓練管線 (Device: {DEVICE})...")

    # 初始化模型設定，配合測試稍降維度
    config = GPTConfig(
        n_layer=6,
        n_head=8,
        n_kv_head=8,
        n_embd=512,
        vocab_size=1000,
        sequence_len=128
    )

    # 1. 宣告三大神經網路實體與 KBGANGenerator
    decoder = KGDecoder(config).to(DEVICE)
    critic = RealismCritic(config).to(DEVICE)
    encoder = KGEncoder(config).to(DEVICE)
    kbgan_gen = KBGANGenerator(config).to(DEVICE)

    # 2. 宣告獨立優化器 (Critic 用較低 LR 避免主導)
    opt_decoder = torch.optim.AdamW(decoder.parameters(), lr=LEARNING_RATE)
    opt_critic = torch.optim.AdamW(critic.parameters(), lr=CRITIC_LR)
    opt_encoder = torch.optim.AdamW(encoder.parameters(), lr=LEARNING_RATE)
    opt_kbgan_gen = torch.optim.AdamW(kbgan_gen.parameters(), lr=LEARNING_RATE)

    # 3. 學習率調度器
    sched_decoder = torch.optim.lr_scheduler.CosineAnnealingLR(opt_decoder, T_max=MAX_STEPS)
    sched_critic = torch.optim.lr_scheduler.CosineAnnealingLR(opt_critic, T_max=MAX_STEPS)
    # Encoder: linear decay to 10% of initial LR — 後期穩定，減少震盪
    sched_encoder = torch.optim.lr_scheduler.LinearLR(opt_encoder, start_factor=1.0, end_factor=0.1, total_iters=MAX_STEPS)
    sched_kbgan = torch.optim.lr_scheduler.CosineAnnealingLR(opt_kbgan_gen, T_max=MAX_STEPS)

    # 4. 結構化合成數據源 (KBGAN 洞見：替代純隨機負採樣)
    data_source = StructuredKGData(config, DEVICE)

    decoder.train()
    critic.train()
    encoder.train()
    kbgan_gen.train()

    t0 = time.time()

    # =========================================================================
    # GAN-Style Alternating Training Loop (KBGAN-enhanced)
    # =========================================================================
    for step in range(MAX_STEPS):
        # 取得每一個 step 的結構化資料
        real_text_ids, gt_kg_features, gt_entities, gt_relations = data_source.sample(BATCH_SIZE)

        # ---------------------------------------------------------------------
        # [Phase 1: 訓練判別器 Critic] - BCE + label smoothing
        # Label smoothing 防止 Critic 過於自信 → 梯度消失 → mode collapse
        # ---------------------------------------------------------------------
        opt_critic.zero_grad()

        real_score = critic(real_text_ids)
        loss_critic_real = F.binary_cross_entropy_with_logits(
            real_score, torch.ones_like(real_score) * (1.0 - LABEL_SMOOTH))

        fake_logits = decoder(gt_kg_features, real_text_ids)
        fake_ids = torch.argmax(fake_logits, dim=-1)
        fake_score = critic(fake_ids.detach())
        loss_critic_fake = F.binary_cross_entropy_with_logits(
            fake_score, torch.zeros_like(fake_score) + LABEL_SMOOTH)

        loss_critic = loss_critic_real + loss_critic_fake
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
        opt_critic.step()

        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # [Phase 2: 訓練生成器 Decoder] - L_realism (騙過 Critic)
        # 使用 Gumbel-Softmax + WGAN loss (minimize -E[critic(fake)])
        # ---------------------------------------------------------------------
        opt_decoder.zero_grad()

        fake_logits_train = decoder(gt_kg_features, real_text_ids)
        # Gumbel-Softmax: hard=True 保持離散化但允許 STE 梯度回流
        fake_soft = F.gumbel_softmax(fake_logits_train, tau=1.0, hard=True, dim=-1)
        # Critic 接受 soft embeddings: (B, T, V) @ (V, n_embd) → (B, T, n_embd)
        fake_embedded = fake_soft @ critic.wte.weight
        fake_pooled = fake_embedded.mean(dim=1)
        fake_score_for_gen = critic.encoder(fake_pooled)

        # Generator 目標：讓 Critic 以為是真的 (target = 1)
        loss_realism = F.binary_cross_entropy_with_logits(
            fake_score_for_gen, torch.ones_like(fake_score_for_gen))
        loss_realism.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), MAX_GRAD_NORM)
        opt_decoder.step()

        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # [Phase 3: 訓練組譯器 Encoder] - L_rec (對比損失)
        # ---------------------------------------------------------------------
        opt_encoder.zero_grad()

        synth_text_ids = torch.argmax(decoder(gt_kg_features, real_text_ids).detach(), dim=-1)
        ent_logits, rel_logits = encoder(synth_text_ids)

        # KBGAN Generator 產生困難負樣本 (Hard Negatives) + REINFORCE log_probs
        hard_neg_ent, hard_neg_rel, _, _ = kbgan_gen(gt_entities, gt_relations)

        # Margin-Based Contrastive Loss
        dist_pos_ent = F.mse_loss(ent_logits, gt_entities, reduction='none').mean(dim=-1)
        dist_neg_ent = F.mse_loss(ent_logits, hard_neg_ent.detach(), reduction='none').mean(dim=-1)
        loss_rec_ent = F.relu(dist_pos_ent - dist_neg_ent + 0.3).mean()

        dist_pos_rel = F.mse_loss(rel_logits, gt_relations, reduction='none').mean(dim=-1)
        dist_neg_rel = F.mse_loss(rel_logits, hard_neg_rel.detach(), reduction='none').mean(dim=-1)
        loss_rec_rel = F.relu(dist_pos_rel - dist_neg_rel + 0.3).mean()

        loss_rec = loss_rec_ent + loss_rec_rel
        loss_rec.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), MAX_GRAD_NORM)
        opt_encoder.step()

        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # [Phase 4: 訓練 KBGAN Generator] - REINFORCE 策略梯度
        # Reward = Encoder 對假樣本的困惑程度 (越高越好 → 成功欺騙 Encoder)
        # ---------------------------------------------------------------------
        opt_kbgan_gen.zero_grad()

        hard_neg_ent_r, hard_neg_rel_r, ent_log_prob, rel_log_prob = kbgan_gen(gt_entities, gt_relations)

        with torch.no_grad():
            # Reward: 假樣本與 Encoder 輸出的距離 (越接近 → 越難分辨 → 越高 reward)
            reward_ent = -F.mse_loss(ent_logits.detach(), hard_neg_ent_r, reduction='none').mean(dim=-1).mean()
            reward_rel = -F.mse_loss(rel_logits.detach(), hard_neg_rel_r, reduction='none').mean(dim=-1).mean()

        # REINFORCE: loss = -log_prob * reward (梯度方向 = 增加高 reward 動作的概率)
        loss_kbgan = -(ent_log_prob * reward_ent + rel_log_prob * reward_rel)
        loss_kbgan.backward()
        torch.nn.utils.clip_grad_norm_(kbgan_gen.parameters(), MAX_GRAD_NORM)
        opt_kbgan_gen.step()

        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # 更新學習率
        sched_decoder.step()
        sched_critic.step()
        sched_encoder.step()
        sched_kbgan.step()

        # ---------------------------------------------------------------------
        # Logging & Debugging
        # ---------------------------------------------------------------------
        if step % 10 == 0 or step == MAX_STEPS - 1:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"[Step {step:03d}] L_critic: {loss_critic.item():.4f} | L_realism: {loss_realism.item():.4f} | L_rec: {loss_rec.item():.4f} | Time: {dt*1000:.2f}ms")

if __name__ == "__main__":
    train()
