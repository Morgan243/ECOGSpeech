from typing import Optional, Tuple, List
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import Module, Parameter

# https://github.com/pytorch/audio/blob/a92ae3688afad51245d135a3f361fb7e20364d6d/torchaudio/models/wav2vec2/components.py#L718
def _compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> Tensor:
    """Computes random mask spans for a given shape.
    Args:
        shape (int, int): The shape for which to compute masks.
            The first element is batch size and second is the number of frames.
        padding_mask (Tensor or None): The padding mask of the same dimension as shape,
            which will prevent masking padded elements.
        mask_prob (float): Probability for each token to be chosen as start of the span to be masked.
            This will be multiplied by number of timesteps divided by length of mask span to mask
            approximately this percentage of all elements. However due to overlaps, the actual number
            will be smaller (unless no_overlap is True).
        mask_type (str): How to compute mask lengths. Options: [``static``, ``uniform``, ``normal``, ``poisson``].
            ``static``: Fixed size
            ``uniform``: Sample from uniform distribution [mask_other, mask_length*2]
            ``normal``: Sample from normal distribution with mean ``mask_length`` and stdev ``mask_other``.
            ``poisson``: Sample from possion distribution with lambda = ``mask_length``.
        min_masks (int): Minimum number of masked spans.
        no_overlap (bool): If false, will switch to an alternative recursive algorithm
            that prevents spans from overlapping.
        min_space (int): How many frames to keep unmasked between spans (Only used if no_overlap is True).
    Returns:
        (Tensor): The mask indices of dimension `[batch, frame]`.
    """

    batch_size, frame = shape
    mask = torch.full((batch_size, frame), False)
    # add a random number for probabilistic rounding
    all_num_mask = int(mask_prob * frame / float(mask_length) + torch.rand(1))

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(batch_size):
        if padding_mask is not None:
            sz = frame - padding_mask[i].long().sum().item()
            # add a random number for probabilistic rounding
            num_mask = int(mask_prob * sz / float(mask_length) + torch.rand(1))
            num_mask = max(min_masks, num_mask)
        else:
            sz = frame
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = torch.full((num_mask,), mask_length)
        elif mask_type == "uniform":
            lengths = torch.randint(mask_other, mask_length * 2 + 1, size=(num_mask,))
        elif mask_type == "normal":
            lengths = torch.normal(mask_length, mask_other, size=(num_mask,))
            lengths = torch.maximum(torch.ones(1), torch.round(lengths)).int()
        elif mask_type == "poisson":
            lengths = torch.poisson(mask_length, size=(num_mask,))
            lengths = torch.round(lengths).int()
        else:
            raise Exception(f"unknown mask selection: {mask_type}")

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = torch.randint(s, e - length, size=(1,))
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = torch.tensor([e - s for s, e in parts], dtype=torch.int)
                lens[lens < length + min_space] = 0
                l_sum = lens.sum()
                if l_sum == 0:
                    break
                probs = lens / l_sum
                c = torch.distributions.categorical.Categorical(probs).sample()
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = torch.tensor(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = torch.multinomial(torch.ones((sz - min_len,)), num_samples=num_mask, replacement=False)

            mask_idc = torch.tensor(
                [mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])]
            )

        mask_idcs.append(torch.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = torch.index_select(
                mask_idc,
                0,
                torch.multinomial(
                    torch.ones((mask_idc.shape[0],)),
                    num_samples=min_len,
                    replacement=False,
                ),
            )
        mask[i, mask_idc] = True

    return mask


class CoG2Vec(torch.nn.Module):
    logit_temp = 0.1

    def __init__(self, input_shape, feature_model, context_model, projection_model, dropout=0.5):
        super().__init__()
        self.input_shape = input_shape
        self.feature_model = feature_model

        # TODO: Don't assign test data as attribute? But it's so useful
        self.t_x = torch.rand((16, *self.input_shape))

        # ####
        # Feature extractor: typically CNN that downsamples/aggregates input for the context and quantize stages
        # Create a default feature exractor model if not provided
        if self.feature_model is None:
            #from fairseq
            # TODO: check on existing norm and GELU activation?
            self.feature_model = torch.nn.Sequential(
                # torch.nn.Conv1d((input_channels:=X_barr.shape[1]), input_channels, 10, stride=5),
                torch.nn.Conv1d((input_channels := 1), (h_channels := 512), 10, stride=5),
                torch.nn.Dropout(p=dropout),
                torch.nn.GELU(),
                torch.nn.Conv1d(h_channels, h_channels, 5, stride=3),
                torch.nn.Dropout(p=dropout),
                torch.nn.GELU(),
                #torch.nn.Conv1d(h_channels, h_channels, 5, stride=3),
                #torch.nn.Conv1d(h_channels, h_channels, 5, stride=2)
            )

        # Run test data through to get sizes automatically
        self.t_feat_o = self.feature_model(self.t_x)

        # for unseen regions
        self.mask_embedding = torch.nn.Parameter(
            torch.FloatTensor(embed_dim := self.t_feat_o.shape[1]).uniform_()
        )
        # For start of sentence - is this needed?
        self.sos_embedding = torch.nn.Parameter(
            torch.FloatTensor(embed_dim).uniform_()
        )
        # For end of sentence - is this needed?
        self.eos_embedding = torch.nn.Parameter(
            torch.FloatTensor(embed_dim).uniform_()
        )

        # Unused, but maybe useful for debugging and experiments
        _, self.C, self.T = self.t_feat_o.shape

        # TODO: Way to override positional part of transformer? Need to integrate xyz eventually
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=h_channels, nhead=8, batch_first=True)
        transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.context_model = transformer_encoder

        # Use existing Gumbel Quant
        import fairseq
        self.quantizer = fairseq.modules.GumbelVectorQuantizer(
            # TODO: parameterize mor of these?
            dim=h_channels, num_vars=320, temp=(1, 0.1, 0.9), groups=2, combine_groups=True, vq_dim=h_channels,
            time_first=False
        )

        # Currently unused, but in future may need one or more linear projections from one space to another
        self.projection_model = torch.nn.Linear(self.t_feat_o.shape[-1], self.t_feat_o.shape[-1])

    # Adapted From fairseq wave2vec2 (remove xla check
    def compute_preds(self, x, y, negatives, ):
        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1)
        logits = logits / self.logit_temp
        logits = logits.type_as(x)

        if neg_is_pos.any():
            logits[neg_is_pos] = float("-inf")
            #logits[1:] = index_put(logits[1:], neg_is_pos, float("-inf"))

        return logits

    # Adapted From fairseq wave2vec2 (remove xla check, add params, buffered arange inline def)
    def sample_negatives(self, y, num, n_negatives, cross_sample_negatives, padding_count=None):
        def buffered_arange(max):
            if not hasattr(buffered_arange, "buf"):
                buffered_arange.buf = torch.LongTensor()
            if max > buffered_arange.buf.numel():
                buffered_arange.buf.resize_(max)
                torch.arange(max, out=buffered_arange.buf)
            return buffered_arange.buf[:max]

        if n_negatives == 0 and cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz, tsz, fsz}"

            if n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                        .unsqueeze(-1)
                        .expand(-1, n_negatives)
                        .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                        .unsqueeze(-1)
                        .expand(-1, cross_sample_negatives)
                        .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if n_negatives > 0:
            neg_idxs = neg_idxs + (torch.arange(bsz).unsqueeze(1) * high)
        else:
            neg_idxs = cross_neg_idxs

        if cross_sample_negatives > 0 and n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, n_negatives + cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def forward(self, X):
        # TODO: Wave2vec says normalize the raw waveform to zero mean and unit variance - maje sure this happening?
        # Extract features from signal
        X_f = self.feature_model(X)
        # penalize for large features
        features_pen = X_f.float().pow(2).mean()

        # Note expected dims: (B)atch, (C)hannel, (T)ime
        B, C, T = X_f.shape

        # Create the mask
        mask_ixes = _compute_mask_indices((B, T), padding_mask=None, mask_prob=0.2, mask_length=1)

        # Create inverse of mask to select unmasked values
        umask_ixes = ~mask_ixes

        # Swap C and T  for use with mask and later transformer sequence modeling
        umasked_X_f = X_f.transpose(1, 2)#.permute(0, 2, 1)

        # Select the masked elements as our y, and reshape back
        _y = umasked_X_f[mask_ixes].view(umasked_X_f.shape[0], -1, umasked_X_f.shape[-1])

        # Go ahead and make a copy of the original data (Same move made in Wave2vec2.py @ line 1021)
        masked_X_f = torch.clone(umasked_X_f)

        # overwrite masked indices with the learnable mask embedding
        masked_X_f[mask_ixes] = self.mask_embedding

        # Quantize the representation
        masked_X_f_c = self.context_model(masked_X_f)

        # Swap time last to last dimension
        # x = masked_X_f_c.permute(0, 2, 1)
        x = masked_X_f_c


        # #################################
        # Code block from fairseq wave2vec2
        negatives_from_everywhere = False
        unmasked_features = umasked_X_f
        mask_indices = mask_ixes
        padding_count = 0
        # Swap the original masked data back to C, T - contiguous call is due to limitation with view() or reshape()
        # in torch tensors
        y = _y.transpose(1, 2).contiguous()
        n_negatives = 100
        cross_sample_negatives = 0
        codebook_negatives = 0  # 100 # This doesn't work..

        if self.quantizer:
            if negatives_from_everywhere:
                # Don't know if this works
                q = self.quantizer(unmasked_features, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]
                # y = self.project_q(y)

                negs, _ = self.sample_negatives(
                    y,
                    mask_indices[0].sum(),
                    padding_count=padding_count,
                )
                y = y[mask_indices].view(y.size(0), -1, y.size(-1))

            else:
                q = self.quantizer(y, produce_targets=False)
                y = q["x"].contiguous()
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]

                # y = self.project_q(y)

                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    n_negatives=n_negatives, cross_sample_negatives=cross_sample_negatives,
                    padding_count=padding_count,
                )

            if codebook_negatives > 0:
                raise NotImplementedError
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), codebook_negatives
                )
                cb_negs = cb_negs.view(
                    codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                # cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            # TODO: project q won't work
            raise NotImplementedError()
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(
                    unmasked_features,
                    y.size(1),
                    n_negatives=n_negatives, cross_sample_negatives=cross_sample_negatives,
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    n_negatives=n_negatives, cross_sample_negatives=cross_sample_negatives,
                    padding_count=padding_count,
                )

        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        y, negs = y.transpose(1, 2), negs.transpose(2, 3)
        preds = self.compute_preds(x, y, negs)

        return dict(x=x, y=y, negs=negs, preds=preds,
                    features_pen=features_pen, num_vars=num_vars,
                    code_perplexity=code_ppl, prob_perplexity=prob_ppl,
                    temp=curr_temp)
        #return x, y, negs, preds

    # Previous/early iteration for reference - will remove
    def forward_old(self, X):
        # TODO: Wave2vec says normalize the raw waveform to zero mean and unit variance - maje sure this happening?
        X_f = self.feature_model(X)
        B, C, T = X_f.shape
        mask_ixes = _compute_mask_indices((B, T), None, 0.2, 1)
        umask_ixes = ~mask_ixes

        umasked_X_f = X_f.permute(0, 2, 1)
        masked_X_f = torch.clone(umasked_X_f)
        # See torchaudio's mask generator that creates a mask and replaces that with parameters:
        #  https://github.com/pytorch/audio/blob/a92ae3688afad51245d135a3f361fb7e20364d6d/torchaudio/models/wav2vec2/components.py#L938
        # This is used in hubert pretrained model forward pass brefore being passed to the transformer (context)
        # There are no wave2vec pretrained models, but presumably would be used there are well?
        masked_X_f[mask_ixes] = self.mask_embedding

        # Quantize the representation
        masked_X_f_c = self.context_model(masked_X_f)
        umasked_X_f_q = self.quantizer(umasked_X_f)

        _X_f = masked_X_f.permute(0, 2, 1)

        # TODO: What? What should be the forward pass here?
        #trf_out = self.context_model.forward(_X_f, _X_f)

        # Create "logits" when there are labels
        proj_x = self.projection_model(masked_X_f_c)
        # - TODO: select positive ("pos") samples in X, select negative ("neg") as all samples in X, measure distance

        m_proj_x = proj_x.permute(0, 2, 1)[mask_ixes].permute(0, 2, 1)
        um_proj_x = proj_x[umask_ixes]

        return umasked_X_f, masked_X_f, _X_f, proj_x, m_proj_x, um_proj_x
        #return mask_ixes, _X_f, trf_out, proj_x

from ecog_speech.models.base import Trainer

class Cog2VecTrainer(Trainer):
    def train_inner_step(self, epoch_i, data_batch):
        res_d = dict()

        model = self.model_map['model']
        optim = self.opt_map['model']
        model = model.train()

        model.zero_grad()
        optim.zero_grad()

        X_barr = data_batch['signal_arr'].to(self.device)
        # Select a single sensor for now and remove the singleton dimension
        X = X_barr.select(1, np.random.randint(0, X_barr.shape[1])).unsqueeze(1)

        m_d = model(X)
        x = m_d['x']
        target = x.new_zeros((x.size(1) * x.size(2), x.size(0)), dtype=torch.long)
        logits = m_d["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))

        loss = F.binary_cross_entropy_with_logits(
            logits, target.float(), #weights, reduction=reduction
        )

        ppl_l = ((m_d["num_vars"] - m_d["prob_perplexity"]) / m_d["num_vars"]) * 0.1
        fpen_l = m_d["features_pen"] * 10
        total_loss = loss + ppl_l + fpen_l
        total_loss.backward()
        optim.step()
        model = model.eval()

        return dict(total_loss=total_loss.detach().cpu().item(),
                    bce_logits=loss.detach().cpu().item(),
                    perplexity=ppl_l.detach().cpu().item(),
                    feature=fpen_l.detach().cpu().item())
