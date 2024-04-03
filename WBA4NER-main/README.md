# WBA4NER


This is the code for the paper [WBA: Word Boundary Attention for Chinese Named Entity Recognition]. 

## Introduction

We advocate a novel word boundary attention enhancement method, Wordformer, that attends word boundaries in a sentence with relative positional encoding of word parallel structure.

## Environment Requirement
The code has been tested under Python 3.7. The required packages are as follows:
```
torch==1.5.1
numpy==1.18.5
FastNLP==0.5.0
fitlog==0.3.2
```
you can click [here](https://fastnlp.readthedocs.io/zh/latest/) to know more about FastNLP. And you can click [here](https://fitlog.readthedocs.io/zh/latest/) to know more about Fitlog.

## Example to Run the Codes
1. Download the pretrained character embeddings and word embeddings and put them in the data folder.
    * Character embeddings (gigaword_chn.all.a2b.uni.ite50.vec): [Google Drive](https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)
    * Bi-gram embeddings (gigaword_chn.all.a2b.bi.ite50.vec): [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)
    * Word(Lattice) embeddings (ctb.50d.vec): [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)
    * If you want to use a larger word embedding, you can refer to [Chinese Word Vectors 中文词向量](https://github.com/Embedding/Chinese-Word-Vectors) and [Tencent AI Lab Embedding](https://ai.tencent.com/ailab/nlp/en/embedding.html)

2. Modify the `utils/paths.py` to add the pretrained embedding and the dataset.

3. Long sentence clipping for MSRA and Ontonotes, run the command:
```bash
python sentence_clip.py
```

4. Merging char embeddings and word embeddings:
```bash
python char_word_mix.py
```

5. Model training and evaluation
    * Weibo dataset
    ```shell
    python main.py --dataset weibo
    ```
    * Resume dataset
    ```shell
    python main.py --dataset resume
    ```
    * Ontonotes dataset
    ```shell
    python main.py --dataset ontonotes
    ```
    * MSRA dataset
    ```shell
    python main.py --dataset msra
    ```

## Acknowledgements
* Thanks to Dr. Li and his team for contributing the [FLAT source code](https://github.com/LeeSureman/Flat-Lattice-Transformer).
* Thanks to the author team and contributors of [TENER source code](https://github.com/fastnlp/TENER).
* Thanks to the author team and contributors of [FastNLP](https://github.com/fastnlp/fastNLP).
