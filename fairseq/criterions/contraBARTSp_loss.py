# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch.nn as nn

@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "contraBARTSp_loss", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class contraBARTSp_loss(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.final_loss = nn.MarginRankingLoss(0)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        
        positive_sample = sample.copy()
        positive_sample.pop('negtive')
        positive_sample["net_input"]['prev_output_tokens'] = sample['positive_prev_output_tokens']
        net_output = model(**positive_sample["net_input"])
        #net_output = model(**sample["net_input"])
        loss1, nll_loss1 = self.compute_loss(model, net_output, sample, reduce=False,computing_pair_target = 'positive')
        
        negtive_sample = sample.copy()
        negtive_sample.pop('negtive')
        negtive_sample["net_input"]['prev_output_tokens'] = sample['negtive_prev_output_tokens']
        net_output = model(**negtive_sample["net_input"])
        loss2, nll_loss2 = self.compute_loss(model, net_output, sample, reduce=False,computing_pair_target = 'negtive')

        pad_mask = sample['positive'].eq(self.padding_idx)
        pad_mask = ~pad_mask + 0
        pos_len = pad_mask.sum(dim = 1)

        pad_mask = sample['negtive'].eq(self.padding_idx)
        pad_mask = ~pad_mask + 0
        neg_len = pad_mask.sum(dim = 1)

        BATCHSIZE = sample["positive"].size(0)
        loss1 = loss1.reshape(BATCHSIZE,-1).sum(dim = 1)
        loss2 = loss2.reshape(BATCHSIZE,-1).sum(dim = 1)
        ones = torch.ones(loss1.size()).cuda(loss1.device)

        #final_loss = self.final_loss(loss2,loss1,ones) + loss1.mean() + loss2.mean()
        final_loss = self.final_loss(loss2,loss1,ones) + loss1.mean()
        #nll_loss = nll_loss1.mean() + nll_loss2.mean()
        nll_loss = nll_loss1.mean()
        sample_size = (
            sample["positive"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": final_loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["positive"].size(0),
            "sample_size": sample_size,
        }

        return final_loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample, computing_pair_target = 'positive'):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        if computing_pair_target == 'positive':
            target = sample["positive"]
        else:
            target = sample["negtive"]
            #target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True, computing_pair_target = 'positive'):
        #reduce = False
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample, computing_pair_target)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
