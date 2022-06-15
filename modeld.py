import flair
from flair.data import Sentence
from flair.embeddings import (FlairEmbeddings, DocumentPoolEmbeddings,
                              ELMoEmbeddings, BertEmbeddings,
                              TransformerDocumentEmbeddings,
                              DocumentRNNEmbeddings)

FlairEmbeddings = FlairEmbeddings("multi-forward", chars_per_chunk=128)  # .flair
FlairEmbeddings = FlairEmbeddings("multi-backward", chars_per_chunk=128)
BertEmbeddings = BertEmbeddings('bert-base-multilingual-cased') # 下载到usr/.cache/huggingface/transformers
ELMoEmbeddings = ELMoEmbeddings('small') # pip install allennlp==0.9.0 # 下载到usr/.allennlp
TransformerDocumentEmbeddings = TransformerDocumentEmbeddings('roberta-base') # 下载到usr/.cache/huggingface/transformers 将SSR代理模式修改为PAC模式