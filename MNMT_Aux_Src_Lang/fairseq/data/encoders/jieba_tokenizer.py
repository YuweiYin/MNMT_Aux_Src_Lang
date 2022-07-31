# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.data.encoders import register_tokenizer
import re
import jieba

@register_tokenizer('jieba')
class JiebaTokenizer(object):

    def __init__(self, args):
        self.args = args
        pattern1 = re.compile(r'([^a-z^A-Z^0-9]) ')
        pattern2 = re.compile(r' ([^a-z^A-Z^0-9])')
        self.detokenizer = lambda s: pattern2.sub(r'\1', pattern1.sub(r'\1', s))

    def encode(self, x: str) -> str:
        return ' '.join(jieba.cut(x)).replace('   ', ' ')

    def decode(self, x: str) -> str:
        return self.detokenizer(' '.join(x.split()))
