import copy
from typing import Optional, Tuple, List

import pandas as pd
from ecog_speech.models import base
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import math
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


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

from fairseq.modules import GradMultiply
class CoG2Vec(torch.nn.Module):
    logit_temp = 0.1

    def __init__(self, input_shape, feature_model, context_model, projection_model, dropout=0.1, feature_grad_mult=1,
                 negatives_from_everywhere=True, n_negatives=100,
                 cross_sample_negatives=0, codebook_negatives=0,
                 mask_length=2, mask_prob=0.2,
                 n_encoder_heads=8, n_encoder_layers=12,
                 quant_num_vars=300, quant_num_groups=2,
                 quant_weight_proj_factor=2, quant_weight_proj_depth=1,
                 feature_extractor_layers='[(128, 7, 3)] + [(128, 3, 2)] * 2 + [(256, 3, 1)]',
                 feature_extractor_mode='layer_norm',
                 squeeze_first=True):
        super().__init__()
        self.input_shape = input_shape
        self.feature_model = feature_model
        self.feature_grad_mult = feature_grad_mult
        self.negatives_from_everywhere = negatives_from_everywhere

        self.n_negatives = n_negatives
        self.cross_sample_negatives = cross_sample_negatives
        self.codebook_negatives = codebook_negatives

        self.mask_length, self.mask_prob = mask_length, mask_prob

        # TODO: Don't assign test data as attribute? But it's so useful
        self.t_x = torch.rand((16, *self.input_shape))

        # ####
        # Feature extractor: typically CNN that downsamples/aggregates input for the context and quantize stages
        # Create a default feature exractor model if not provided
        from fairseq.models.wav2vec import ConvFeatureExtractionModel
        self.squeeze_first = squeeze_first

        if self.feature_model is None:
            conv_layers = (eval(feature_extractor_layers) if isinstance(feature_extractor_layers, str)
                           else feature_extractor_layers)
            #h_channels = 256
            self.feature_model = ConvFeatureExtractionModel(
                #conv_layers=[(128, 7, 3)] + [(128, 3, 2)] * 2 + [(f_dim, 3, 1)],
                conv_layers=conv_layers,
                dropout=dropout,
                mode=feature_extractor_mode,#'layer_norm',
                # mode="default",#cfg.extractor_mode,
                conv_bias=False  # cfg.conv_bias,
            )
            #from fairseq
            # TODO: check on existing norm and GELU activation?
#            self.feature_model = torch.nn.Sequential(
#                # torch.nn.Conv1d((input_channels:=X_barr.shape[1]), input_channels, 10, stride=5),
#                #torch.nn.BatchNorm1d(1),
#                torch.nn.Conv1d((input_channels := 1), (h_channels := 256), 7, stride=5),
#                torch.nn.Dropout(p=dropout),
#                torch.nn.GELU(),
#
#                torch.nn.Conv1d(h_channels, h_channels, 5, stride=2),
#                #torch.nn.BatchNorm1d(h_channels),
#                #torch.nn.Dropout(p=dropout),
#                torch.nn.GELU(),
#
#                torch.nn.Conv1d(h_channels, (f_dim:=128), 3, stride=1),
#                #torch.nn.Dropout(p=dropout),
#                torch.nn.GELU(),
#            )
#            self.feature_model.apply(base.weights_init)


        # Run test data through to get sizes automatically
        with torch.no_grad():
            self.t_feat_o = self.feature_model(self.t_x.squeeze() if self.squeeze_first else self.t_x)

        # Unused, but maybe useful for debugging and experiments
        _, self.C, self.T = self.t_feat_o.shape
        embed_dim = self.C

        self.feature_norm = torch.nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True)

        # for unseen regions
        self.mask_embedding = torch.nn.Parameter(
            torch.FloatTensor(embed_dim).uniform_()
        )
        # For start of sentence - is this needed?
        self.sos_embedding = torch.nn.Parameter(
            torch.FloatTensor(embed_dim).uniform_()
        )
        # For end of sentence - is this needed?
        self.eos_embedding = torch.nn.Parameter(
            torch.FloatTensor(embed_dim).uniform_()
        )

        self.positional_enc = PositionalEncoding(d_model=embed_dim)

        # TODO: Way to override positional part of transformer? Need to integrate xyz eventually
        self.n_heads = n_encoder_heads
        self.num_encoder_layers = n_encoder_layers
        self.context_model = context_model
        if self.context_model is None:
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=self.n_heads, batch_first=True,
                                                             activation="gelu")
            transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)
            self.context_model = transformer_encoder

        self.quant_weight_proj_depth, self.quant_weight_proj_factor = quant_weight_proj_depth, quant_weight_proj_factor
        # Use existing Gumbel Quant
        import fairseq
        self.quantizer = fairseq.modules.GumbelVectorQuantizer(
            # TODO: parameterize mor of these?
            dim=embed_dim, num_vars=quant_num_vars, temp=(2, 0.5, 0.999995),#temp=(1, 0.1, 0.9),
            groups=quant_num_groups, combine_groups=False, vq_dim=embed_dim,
            time_first=True,
            # defaults in fairseq config
            weight_proj_factor=self.quant_weight_proj_factor, weight_proj_depth=self.quant_weight_proj_depth
        )

        # Currently unused, but in future may need one or more linear projections from one space to another
        self.projection_q_model = projection_model

        if self.projection_q_model is None:
            self.projection_q_model = torch.nn.Linear(embed_dim, embed_dim)

        self.projection_out_model = None
        if self.projection_out_model is None:
            self.projection_out_model = torch.nn.Linear(embed_dim, embed_dim)

    # Adapted From fairseq wave2vec2 (remove xla check
    def compute_preds(self, x, y, negatives, ):
        # Negatives: n_negatives x B x T x C

        # determine where negatives are that are also positives
        # This is at the feature level, so identifying a specific negative,
        # within a sample in the batch, at specific time tgat is actually a positive
        neg_is_pos = (y == negatives).all(-1)
        # Add the first dimension to match with negatives addition first dim of n_negatives size
        y = y.unsqueeze(0)
        # Combine these on the first dim - actual values now appear as a +1 to the negatives
        # but always at the first (0th) dimension
        targets = torch.cat([y, negatives], dim=0)

        # Measure context models output similarity the K+1 targets
        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1)
        logits = logits / self.logit_temp
        logits = logits.type_as(x)

        if neg_is_pos.any():
            # slice at 1, beyond 0th element (the true positive) and set these to negative inf
            # A good model would identify these, but the BCE penalty would penalize those, so
            # just wipe them out with a very low logit
            #logits[1:][neg_is_pos] = float("-inf")
            logits[1:][neg_is_pos] = float(-100)
            # original from fairseq
            #logits[1:] = index_put(logits[1:], neg_is_pos, float("-inf"))

        return logits

    # Adapted From fairseq wave2vec2 (remove xla check, add params, buffered arange inline def)
    @staticmethod
    def sample_negatives(y, num, n_negatives, cross_sample_negatives=0, padding_count=None):
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
        # Flatt to all feature samples (flatten batch and time)
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz, tsz, fsz}"

            if n_negatives > 0:
                # Possible time steps to use, {0, 1, 2, 3, ... num.}  for each negative
                tszs = (
                    # Long Tensor range 0 to NUm - 1
                    buffered_arange(num)
                    # Add a singleton dim at the end
                        .unsqueeze(-1)
                    # Duplicate data across singleton dimension n_negatives times
                    # now (tsz, n_negatives), incrementing over tsz
                        .expand(-1, n_negatives)
                    # like reshape(-1), len == tsc * n_negatives
                        .flatten()
                )

                #
                neg_idxs = torch.randint(
                    # Note that high here is exlusive - it's "one above the highest number to be drawn"
                    # So this is one less than the max index, allowing the shift by one to happen later
                    low=0, high=high - 1,
                    # Each sample in batch needs a set of negatives...for each masked timestamp?
                    size=(bsz, n_negatives * num)
                )
                # Add one in order to index over the true positive at the starting position, and shift all other
                # times from that point. If only shifting the matching, then the point after will be selected twice
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
            # Offset into each batch
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

    def forward(self, X, features_only=False, mask=True,
                ):
        # TODO: Wave2vec says normalize the raw waveform to zero mean and unit variance - maje sure this happening?
        # Extract features from signal
        # Wave to vec in fair seq either does forward without gradient or with grad multiply - prob grad mult for pre-train

        X = X.squeeze() if self.squeeze_first else X
        X_f = self.feature_model(X)
        if self.feature_grad_mult != 1.0:
            X_f = GradMultiply.apply(X_f, self.feature_grad_mult)

        # Note expected dims: (B)atch, (C)hannel, (T)ime
        B, C, T = X_f.shape

        # Swap C and T  for use with mask and later transformer sequence modeling
        unmasked_features = X_f.transpose(1, 2)
        unmasked_features = self.feature_norm(unmasked_features)

        if mask:
            # Create the mask
            mask_indices = _compute_mask_indices((B, T), padding_mask=None, mask_prob=self.mask_prob,
                                                 mask_length=self.mask_length, min_masks=1)

            # Create inverse of mask to select unmasked values
            #umask_ixes = ~mask_indices

            # Select the masked elements as our y, and reshape back
            y = unmasked_features[mask_indices].view(unmasked_features.shape[0], -1, unmasked_features.shape[-1]).contiguous()

            # Go ahead and make a copy of the original data (Same move made in Wave2vec2.py @ line 1021)
            masked_features = torch.clone(unmasked_features)

            # overwrite masked indices with the learnable mask embedding
            masked_features[mask_indices] = self.mask_embedding
            in_to_context = masked_features
        else:
            mask_indices = None
            in_to_context = unmasked_features
            y = unmasked_features

        # Optionally positionally encode
        if self.positional_enc is not None:
            in_to_context = self.positional_enc(in_to_context)

        x = self.context_model(in_to_context)

        # #################################
        # Code block from fairseq wave2vec2
        if features_only:
            d = {
                "x": x,
                #"padding_mask": padding_mask,
                "features": unmasked_features,
                #"layer_results": layer_results,
            }
            if mask:
                d['mask_indices'] = mask_indices
                d['mask_features'] = in_to_context
                d['y'] = y
            return d

        # penalize for large features
        features_pen = X_f.float().pow(2).mean()

        padding_count = 0
        # Swap the original masked data back to C, T - contiguous call is due to limitation with view() or reshape()
        # in torch tensors
        #n_negatives = 100
        #cross_sample_negatives = 20
        #codebook_negatives = 20  # 100 # This doesn't work..

        if self.quantizer:
            if self.negatives_from_everywhere:
                # Don't know if this works
                q = self.quantizer(unmasked_features, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]

                y = self.projection_q_model(y)

                negs, _ = self.sample_negatives(
                    y,
                    mask_indices[0].sum(),
                    padding_count=padding_count, cross_sample_negatives=self.cross_sample_negatives,
                    n_negatives=self.n_negatives
                )
                y = y[mask_indices].view(y.size(0), -1, y.size(-1))

            else:
                q = self.quantizer(y, produce_targets=False)
                y = q["x"].contiguous()
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]

                y = self.projection_q_model(y)

                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    n_negatives=self.n_negatives, cross_sample_negatives=self.cross_sample_negatives,
                    padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                #raise NotImplementedError
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
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

        # Get outputs from the context model for masked regions - it's estimatino of the quantized output
        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        #
        x = self.projection_out_model(x)
        preds = self.compute_preds(x, y, negs)

        return dict(x=x, y=y, negs=negs, preds=preds,
                    mask_indices=mask_indices,
                    unmasked_features=unmasked_features,
                    features_pen=features_pen, num_vars=num_vars,
                    code_perplexity=code_ppl, prob_perplexity=prob_ppl,
                    temp=curr_temp)

    def create_fine_tuning_model(self, options=None, classifier_head=None):
        from ecog_speech.models import base

        # Very simple linear classifier head by default
        if classifier_head is None:
            classifier_head = torch.nn.Sequential(*[
                base.Reshape((self.C * self.T,)),
                torch.nn.Linear(self.C * self.T, 1),
                torch.nn.Sigmoid()
            ])

        m = copy.deepcopy(self)
        m.feature_model.requires_grad_(False)
        m.quantizer.codebook_indices = None

        ft_model = FineTuner(pre_trained_model=m, output_model=classifier_head,
                             pre_trained_model_output_key='x')
        return ft_model


class FineTuner(torch.nn.Module):
    def __init__(self, pre_trained_model, output_model, pre_trained_model_output_key):
        super().__init__()
        self.pre_trained_model = pre_trained_model
        self.output_model = output_model
        self.pre_trained_model_output_key = pre_trained_model_output_key

    def forward(self, x):
        pt_out = self.pre_trained_model(x)
        return self.output_model(pt_out[self.pre_trained_model_output_key])

from ecog_speech.models.base import Trainer
import attr
from tqdm.auto import tqdm
@attr.s
class Cog2VecTrainer(Trainer):
    input_key = attr.ib('signal_arr')
    squeeze_first = True
    #model_output_logits_key = 'x'
    model_output_logits_key = 'preds'
    ppl_weight = 100.

    def _score(self, epoch_i, dataloader, model_key='model'):
        model = self.model_map[model_key]
        model.eval()

        self.best_cv = getattr(self, 'best_cv', np.inf)

        preds_l, actuals_l, loss_l = list(), list(), list()
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc="Eval") as pbar:
                for i, _x in enumerate(dataloader):
                    _X = _x['signal_arr'].to(self.device)
                    _b_X = _X.select(1, sens_id := 10).unsqueeze(1)
                    out_d = model(_b_X, mask=False, features_only=True)
                    y = cog2vec.quantizer(out_d['features'])['x']
                    scores = torch.cosine_similarity(out_d['x'], y, dim=-1)
                    scores_l = 1 - scores.mean()
                    loss_l.append(scores_l.detach().cpu().item())
                    #preds = model(_x['ecog_arr'].to(self.device))
                    #actuals = _x['text_arr'].to(self.device)
                    #loss = self.criterion(preds, actuals)

                    #loss_l.append(loss.detach().cpu().item())

                    pbar.update(1)

                mean_loss = np.mean(loss_l)
                desc = "Mean Eval Loss: %.5f" % mean_loss
                if self.model_regularizer is not None:
                    reg_l = self.model_regularizer(model)
                    desc += (" (+ %.6f reg loss = %.6f)" % (reg_l, mean_loss + reg_l))
                else:
                    reg_l = 0.
                overall_loss = (mean_loss + reg_l)
                if overall_loss < self.best_cv:

                    self.best_model_state = base.copy_model_state(model)
                    self.best_model_epoch = epoch_i
                    self.best_cv = overall_loss
                    desc += "[[NEW BEST]]"

                pbar.set_description(desc)

        self.model_map['model'].train()
        return loss_l


    def eval_on(self, dataloader, cb=None):
        model = self.model_map['model']
        model.eval()
        eval_results_d_l = list()
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc="Eval") as pbar:
                for i, _x in enumerate(dataloader):
                    X = _x['signal_arr'].to(self.device)

                    if self.squeeze_first:
                        X = X.squeeze()

                    m_d = model(X)

                    loss_d = self.loss(m_d, as_tensor=False)
                    score_d = self.score_training(m_d, as_tensor=False)

                    eval_d = dict(**score_d, **loss_d)
                    eval_results_d_l.append(eval_d)

                    pbar.update(1)

                eval_res_df = pd.DataFrame(eval_results_d_l)#.mean().to_dict()
                if cb is not None:
                    cb(eval_res_df, pbar)

            return eval_res_df

    def _eval(self, epoch_i, dataloader, model_key='model', primary_eval_key='bce_loss'):
        model = self.model_map[model_key]
        model.eval()
        self.best_cv = getattr(self, 'best_cv', np.inf)

        def cb(_eval_res_df, pbar):
            _eval_res_d = _eval_res_df.mean().to_dict()

            desc = ", ".join(f"{k}={np.round(r, 4)}" for k, r in _eval_res_d.items())
            mean_loss = _eval_res_d[primary_eval_key]

            if self.model_regularizer is not None:
                reg_l = self.model_regularizer(model)
                desc += (" (+ %.6f reg loss = %.6f)" % (reg_l, mean_loss + reg_l))
            else:
                reg_l = 0.
            overall_loss = (mean_loss + reg_l)
            if overall_loss < self.best_cv:

                self.best_model_state = base.copy_model_state(model)
                self.best_model_epoch = epoch_i
                self.best_cv = overall_loss
                desc += "[[NEW BEST]]"

            pbar.set_description(desc)

        eval_res_df = self.eval_on(dataloader, cb=cb)
        eval_res_d = eval_res_df.mean().to_dict()
        eval_res_d['primary_loss'] = eval_res_d[primary_eval_key]

        self.model_map['model'].train()

        return eval_res_d


    def loss(self, model_output_d, as_tensor=True):
        logits = model_output_d[self.model_output_logits_key]
        num_vars = model_output_d['num_vars']
        prob_ppl = model_output_d['prob_perplexity']

        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        target = torch.zeros_like(logits)

        loss = F.cross_entropy(
            logits, target[:, 0].long(), reduction='sum' #weights, reduction=reduction
        )
        #loss = F.binary_cross_entropy_with_logits(
        #    logits, target.float(), reduction='sum' #weights, reduction=reduction
        #)

        ppl_l = ((num_vars - prob_ppl) / num_vars) * self.ppl_weight #* 0.1
        fpen_l = model_output_d["features_pen"] #* 10

        o = dict(bce_loss=loss, perplexity=ppl_l, feature_pen=fpen_l)
        if not as_tensor:
            o = {k: v.detach().cpu().item() for k, v in o.items()}
        return o

    def score_training(self, model_output_d, as_tensor=False):
        """training score metrics that don't require gradient, won't have .backward() called,
        compliments loss for understanding performance"""
        logits = model_output_d[self.model_output_logits_key]

        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))

        with torch.no_grad():
            _max = logits.argmax(-1) == 0
            _min = logits.argmin(-1) == 0

            both = _max & _min
            corr = _max.long().sum().item() - both.long().sum().item()
            count = float(_max.numel())

        acc = (corr / count)
        acc = torch.Tensor([acc]) if as_tensor else acc
        count = torch.Tensor([count]) if as_tensor else count
        return dict(accuracy=acc, n=count)

    def train_inner_step(self, epoch_i, data_batch):
        res_d = dict()

        model = self.model_map['model']
        optim = self.opt_map['model']
        model = model.train()

        model.zero_grad()
        optim.zero_grad()

        X_barr = data_batch['signal_arr'].to(self.device)
        bsz = X_barr.shape[0]
        # Select a single sensor for now and remove the singleton dimension
        #X = X_barr.select(1, np.random.randint(0, X_barr.shape[1])).unsqueeze(1)
        X = X_barr#.select(1, np.random.randint(0, X_barr.shape[1])).unsqueeze(1)

        if self.squeeze_first:
            X = X.squeeze()

        m_d = model(X)

        loss_d = self.loss(m_d)

        total_loss = sum((loss_d.values()))
        total_loss.backward()

        optim.step()

        model = model.eval()

        score_d = self.score_training(m_d)

        tl = total_loss.detach().cpu().item()

        return dict(
                    # Unpack score floats or tensors
                    **{s_k: s.detach().cpu().item() if isinstance(s, torch.Tensor) else s
                       for s_k, s in score_d.items()},

                    # Total loss right after scores
                    total_loss=tl,

                    # Unpack losses - should be all Tensors?
                    **{l_k: l.detach().cpu().item() if isinstance(l, torch.Tensor) else l
                       for l_k, l in loss_d.items()}
                    )

    def generate_outputs_from_model_inner_step(self, model, data_batch, criterion=None,
                                               input_key='signal_arr', target_key='text_arr', device=None,
                                               ):
        X = data_batch[input_key].to(device)

        if self.squeeze_first:
            X = X.squeeze()

        with torch.no_grad():
            model.eval()
            model.to(device)
            m_d = model(X)

        loss_d = self.loss(m_d)
        score_d = self.score_training(m_d, as_tensor=True)
        eval_d = dict(**score_d, **loss_d)

        return eval_d


if __name__ == """__main__""":
    # Demo the model and trainer for debug/testing - use an experiments interface directly for a full CLI
    from ecog_speech.experiments import semi_supervised
    from ecog_speech import utils

    options = utils.build_default_options(semi_supervised.ss_option_kwargs,
                                          train_sets='UCSD-22',
                                          device='cpu',
                                          n_epochs=30)

    results = semi_supervised.run(options)
