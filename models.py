
    def dot_similarity(x1, x2):
        return torch.matmul(x1, x2.t())

    class bert_cont_mix(nn.Module):
        def __init__(self, num_label):
            super().__init__()

            self.bert = BertModel.from_pretrained("../bert-base-chinese")

            self.prejector = nn.Sequential(
                nn.Linear(768, 768),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(768, 768),
            )

            self.opi_cls = nn.Sequential(
                nn.Linear(768, num_label)
            )
            self.tua = 0.7

        def Contrast_loss_func(self, ebd1, ebd2):
            ebd1 = self.prejector(ebd1)
            ebd2 = self.prejector(ebd2)

            b, h = ebd1.shape

            l_pos = torch.bmm(ebd1.view(b, 1, h), ebd2.view(b, h, 1)).squeeze(-1)
            l_neg = torch.mm(ebd1, ebd2.T)

            logits = torch.cat([l_pos, l_neg], dim=1)
            labels = torch.zeros(b, dtype=torch.long, device=device)

            loss = F.cross_entropy(logits / self.tua, labels, reduction="none")

            return loss

        def ContrastMix(self, ebd1, ebd2, Mix_ebd, lam, mix_ids):
            ebd1 = self.Aug_ebd(ebd1)
            ebd2 = self.Aug_ebd(ebd2)
            Mix_ebd = self.Aug_ebd(Mix_ebd)

            sample_loss = self.Contrast_loss_func(ebd1, ebd2).mean()

            Mix_loss1 = self.Contrast_loss_func(ebd1, Mix_ebd)
            Mix_loss2 = self.Contrast_loss_func(ebd2[mix_ids], Mix_ebd)

            Mix_loss = (lam * Mix_loss1 + (1 - lam) * Mix_loss2).mean()

            return 0.7 * sample_loss + 0.5 * Mix_loss

        def Aug_ebd(self, ebd):
            is_zero = (torch.sum(torch.abs(ebd), dim=2) > 1e-8).float()
            soft_len = torch.sum(is_zero, dim=1, keepdim=True)
            soft_len[soft_len < 1] = 1
            ebd = torch.sum(ebd, dim=1)
            ebd = ebd / soft_len

            return ebd

        def forward(self, X):
            Embed1 = self.bert.embeddings(input_ids=X["input_ids"])
            Embed2 = self.bert.embeddings(input_ids=X["aug_input_ids"])

            batch_size, seq_len = Embed1.size(0), Embed1.size(1)
            mix_ids = torch.randint(batch_size, (batch_size,))

            Mix_Embed = Embed2[mix_ids]

            lam = torch.distributions.beta.Beta(0.5, 0.5).sample((batch_size, 1, 1)).to(device)
            Mix_Embed = lam * Embed1 + (1 - lam) * Mix_Embed

            ebd1 = self.bert.encoder(
                hidden_states=Embed1,
                attention_mask=X["attention_mask"].unsqueeze(1).unsqueeze(1)
            )[0]
            ebd2 = self.bert.encoder(
                hidden_states=Embed2,
                attention_mask=X["attention_mask"].unsqueeze(1).unsqueeze(1)
            )[0]
            Mix_ebd = self.bert.encoder(
                hidden_states=Mix_Embed,
                attention_mask=X["attention_mask"].unsqueeze(1).unsqueeze(1)
            )[0]

            out_opi = self.opi_cls(ebd1)

            loss_ccont = self.ContrastMix(ebd1, ebd2, Mix_ebd, lam.squeeze(-1), mix_ids)

            return out_opi, loss_ccont