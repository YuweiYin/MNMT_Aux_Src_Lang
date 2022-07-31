# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import OrderedDict
import copy
import itertools
import os
import numpy as np
from argparse import Namespace
import json
import logging

import torch

from fairseq import options, utils, metrics
from fairseq.data import (
    iterators,
    FairseqDataset,
    Dictionary,
    ConcatDataset,
    data_utils,
    encoders
)
from fairseq.data.language_pair_langid_dataset import LanguagePairLangidDataset
from fairseq.data.resampling_dataset import ResamplingDataset
from fairseq.data.sort_dataset import SortDataset
from . import FairseqTask, register_task
from distoptim.torchdist import rank

from fairseq.data.langpair_dataset_loader import (
    LangpairDatasetLoader,
    get_sample_prob,
    get_size_ratio,
)
EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)
LOGLEVEL = logging.INFO if (rank() == 0) else logging.WARNING
logger.setLevel(level=LOGLEVEL)

def _lang_token(lang: str):
    return '__{}__'.format(lang)


def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, \
        'cannot find language token for lang {}'.format(lang)
    return idx


@register_task('translation_mtl_ols')
class TranslationMtlOlsTask(FairseqTask):
    """A task for training multiple translation models simultaneously.

    Different from multilingual_translation task that iterate over different languages
    Each batch consists of randomly sampled sentences from different language pairs
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs: en-de,en-fr,de-fr')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language (only needed for inference)')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language (only needed for inference)')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--encoder-langtok', default='tgt', type=str, choices=['src', 'tgt'],
                            metavar='SRCTGT',
                            help='replace beginning-of-sentence in source sentence with source or target '
                                 'language token. (src/tgt) default to be target lan_id')
        parser.add_argument('--decoder-langtok', action='store_true',
                            help='replace beginning-of-sentence in target sentence with target language token')
        
        # BT
        parser.add_argument('--data-bt', metavar='DIR', default=None, help='path to back translation data directory')
        parser.add_argument('--downsample-bt', action='store_true',
                            help="downsample bt to match the length of bitext data")

        # online sampling
        parser.add_argument('--language-sample-temperature', default=1.0, type=float, 
                            help='sampling temperature for multi-languages')
        parser.add_argument('--language-upsample-max', action='store_true',
                            help='upsample to make the max-capacity language a full set '
                                 '(default: upsample and downsample to maintain the same total corpus size)')

        # dynamic temperature
        parser.add_argument('--language-temperature-scheduler', default='static', type=str, 
                            help='sampling temperature scheduler [static, linear]')
        parser.add_argument('--min-language-sample-temperature', default=1.0, type=float, 
                            help='min (starting) sampling temperature')
        parser.add_argument('--language-sample-warmup-epochs', default=0, type=int, 
                            help='warmup epochs for language sampling scheduler')
        
        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenizer before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='if setting, we compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-tok', type=str, default="zh",
                            help='if eval-tokenized-bleu is true, choose which type of tokenizer')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        # fmt: on


    def __init__(self, args, src_dict, tgt_dict, training, src_token_range=None, tgt_token_range=None):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.src_token_range = src_token_range
        self.tgt_token_range = tgt_token_range

        self.lang_pairs = args.lang_pairs
        # eval_lang_pairs for multilingual translation is usually all of the
        # lang_pairs. However for other multitask settings or when we want to
        # optimize for certain languages we want to use a different subset. Thus
        # the eval_lang_pairs class variable is provided for classes that extend
        # this class.
        self.eval_lang_pairs = args.lang_pairs
        # model_lang_pairs will be used to build encoder-decoder model pairs in
        # models.build_model(). This allows multitask type of sub-class can
        # build models other than the input lang_pairs
        self.model_lang_pairs = copy.copy(args.lang_pairs)
        self.langs = sorted(list({x for lang_pair in args.lang_pairs for x in lang_pair.split('-')}))
        self.training = training

        self.data_lengths = None
        self.dataset_to_epoch_iter = {}
        self.use_bt = False
        self.data_bt_lengths = None

        self.update_iterator = (self.args.language_sample_temperature != 1. \
            or self.args.language_temperature_scheduler != "static" \
            or self.args.downsample_bt 
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        src_dict, tgt_dict, training, src_token_range, tgt_token_range = cls.prepare(args, **kwargs)
        return cls(args, src_dict, tgt_dict, training, src_token_range, tgt_token_range)
    
    @classmethod
    def prepare(cls, args, **kargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        if args.lang_pairs is None:
            raise ValueError('--lang-pairs is required. List all the language pairs in the training objective.')
        args.lang_pairs = args.lang_pairs.split(',')
        sorted_langs = sorted(list({x for lang_pair in args.lang_pairs for x in lang_pair.split('-')}))
        if args.source_lang is not None or args.target_lang is not None:
            training = False
            args.lang_pairs = ['{}-{}'.format(args.source_lang, args.target_lang)]
        else:
            training = True
            args.source_lang, args.target_lang = "src", "tgt"

        # load dictionaries
        paths = args.data.split(':')
        assert len(paths) > 0
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.src.txt'))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.tgt.txt'))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        src_token_range = ((src_dict.nspecial, len(src_dict)))
        tgt_token_range = ((tgt_dict.nspecial, len(tgt_dict)))

        # add language token
        if args.encoder_langtok is not None or args.decoder_langtok:
            for lang_to_add in sorted_langs:
                _land_idx = src_dict.add_symbol(_lang_token(lang_to_add))
                src_dict.add_special_symbol(_land_idx)
                _land_idx = tgt_dict.add_symbol(_lang_token(lang_to_add))
                tgt_dict.add_special_symbol(_land_idx)
        logger.info('| [src] dictionary: {} types, token range: {}'.format(len(src_dict), src_token_range))
        logger.info('| [tgt] dictionary: {} types, token range: {}'.format(len(tgt_dict), tgt_token_range))
        return src_dict, tgt_dict, training, src_token_range, tgt_token_range
    
    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        # only rebuild iterator when online resampling needed
        if not self.update_iterator and dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]   

        seed = seed + epoch
        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        # has to do so for online resampling
        dataset = self.set_translation_epoch(dataset, epoch=epoch, seed=seed)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        indices = data_utils.filter_by_size(
            indices, dataset, max_positions, raise_exception=(not ignore_invalid_inputs),
        )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )
        self.dataset_to_epoch_iter[dataset] = epoch_iter
        return epoch_iter
    
    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        bt_data_path = None
        if self.args.data_bt is not None:
            paths_bt = self.args.data_bt.split(':')
            bt_data_path = paths_bt[epoch % len(paths_bt)]

        language_sample_temperature = getattr(
            self.args, "min_language_sample_temperature", self.args.language_sample_temperature)
        if self.args.language_temperature_scheduler == "static":
            language_sample_temperature = self.args.language_sample_temperature
        resample = language_sample_temperature != 1.0 or self.args.language_temperature_scheduler != "static"
        is_train = (split == 'train')

        from fairseq import meters
        load_timer = meters.StopwatchMeter()
        load_timer.start()

        dataset_loader = LangpairDatasetLoader(
            data_path, split, self.lang_pairs, self.src_dict, self.tgt_dict,
            combine=combine, 
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            encoder_langtok=self.args.encoder_langtok,
            decoder_langtok=self.args.decoder_langtok,
            seed=self.args.seed,
            epoch=epoch,
            resample=(resample and is_train),
            language_sample_temperature=language_sample_temperature,
            language_upsample_max=self.args.language_upsample_max,
            bt_data_path=bt_data_path,
            is_train=is_train,
            downsample_bt=self.args.downsample_bt
        )
        self.datasets[split], data_lengths, bt_lengths = dataset_loader.load_all_langpair_dataset()
        self.data_lengths = data_lengths if is_train else self.data_lengths
        self.data_bt_lengths = bt_lengths if is_train else self.data_bt_lengths

        load_timer.stop()
        logger.info('| loading dataset took total {} seconds'.format(load_timer.sum))
    
    def build_dataset_for_inference(self, src_tokens, src_lengths):
        src_langs = [self.args.source_lang] * len(src_lengths)
        tgt_langs = [self.args.target_lang] * len(src_lengths)
        return LanguagePairLangidDataset(
            src_tokens, src_lengths, self.source_dictionary,
            src_langs, tgt_langs=tgt_langs,
            encoder_langtok=self.args.encoder_langtok,
            decoder_langtok=self.args.decoder_langtok
        )

    def build_model(self, args):
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator(Namespace(**gen_args))
        return super().build_model(args)

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs('_bleu_counts_' + str(i)))
                totals.append(sum_logs('_bleu_totals_' + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_bleu_counts', np.array(counts))
                metrics.log_scalar('_bleu_totals', np.array(totals))
                metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters['_bleu_counts'].sum,
                        total=meters['_bleu_totals'].sum,
                        sys_len=meters['_bleu_sys_len'].sum,
                        ref_len=meters['_bleu_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu)

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                escape_unk=escape_unk,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        tokenize = sacrebleu.DEFAULT_TOKENIZER if not self.args.eval_tokenized_bleu else self.args.eval_bleu_tok
        return sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def get_sampling_temperature(self, epoch):
        if self.args.language_temperature_scheduler == "linear":
            # epoch * (T-T_0)/warmup_epochs + T_0
            t = self.args.language_sample_temperature - self.args.min_language_sample_temperature
            t *= float(epoch) / self.args.language_sample_warmup_epochs
            return t + self.args.min_language_sample_temperature
        raise NotImplementedError

    def get_resampling_size_ratio(self, epoch):
        new_temp = self.get_sampling_temperature(epoch)
        sample_probs = get_sample_prob(self.data_lengths, new_temp)
        size_ratios = get_size_ratio(
            sample_probs, self.data_lengths, 
            language_upsample_max=self.args.language_upsample_max)
        logger.info("| [resample]  epoch {:03d}, sampling temperature: T = {}".format(epoch, new_temp))
        logger.info("| [resample]  epoch {:03d}, sampling probability by language: {}".format(
            epoch, ", ".join(["{}: {:0.4f}".format(_lang, sample_probs[_i])
            for _i, _lang in enumerate(self.lang_pairs)])))
        logger.info("| [resample]  epoch {:03d}, up/down sampling ratio by language: {}".format(
            epoch, ", ".join(["{}: {:0.2f}".format(_lang, size_ratios[_i])
            for _i, _lang in enumerate(self.lang_pairs)])))
        return size_ratios
    
    def set_translation_epoch(self, dataset, seed=1, epoch=0):
        """
        set epoch for translation dataset resampling
        update size ratio if changing sampling temperature
        """
        # initialize the dataset with the correct starting epoch
        # has to do so for online resampling
        if epoch > 0 and self.args.language_temperature_scheduler != "static" and \
            epoch <= self.args.language_sample_warmup_epochs:
            # update epoch and size ratios
            size_ratios = self.get_resampling_size_ratio(epoch)
            if self.args.downsample_bt:
                bt_ratio = min(sum(self.data_lengths) / float(self.data_bt_lengths), 1.0)
                logger.info("[resample] epoch {:03d}, downsampling ratio for bt: {}".format(epoch, bt_ratio))
                size_ratios = np.concatenate([size_ratios, np.array([bt_ratio])])
            dataset.set_epoch(epoch, size_ratios=size_ratios)

            # reset sort order
            if isinstance(dataset, SortDataset):
                with data_utils.numpy_seed(seed + epoch):
                    shuffle = np.random.permutation(len(dataset))
                dataset.set_sort_order(sort_order=[shuffle, dataset.sizes])
        else:
            dataset.set_epoch(epoch)

            if isinstance(dataset, SortDataset):
                with data_utils.numpy_seed(seed + epoch):
                    shuffle = np.random.permutation(len(dataset))
                dataset.set_sort_order(sort_order=[shuffle, dataset.sizes])
        return dataset
