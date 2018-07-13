Pre-trained ELMo Representations for Many Languages
===================================================

We release our ELMo representations trained on many languages
which helps us win the [CoNLL 2018 shared task on Universal Dependencies Parsing](http://universaldependencies.org/conll18/results.html)
according to LAS.

## Technique Details

We use the same hyperparameter settings as [Peters et al. (2018)](https://arxiv.org/abs/1802.05365) for the biLM
and the character CNN.
We train their parameters
on a set of 20-million-words data randomly
sampled from the raw text released by the shared task (wikidump + common crawl) for each language.
We largely based ourselves on the code of [AllenNLP](https://allennlp.org/), but made the following changes:

* We support unicode characters;
* We use the *sample softmax* technique
to make training on large vocabulary feasible ([Jean et al., 2015](https://arxiv.org/abs/1412.2007)).
However, we use a window of words surrounding the target word
as negative samples and it shows better performance in our preliminary experiments.

The training of ELMo on one language takes roughly 3 days on an NVIDIA P100 GPU.


## Downloads

|   |   |   |   |
|---|---|---|---|
| [Arabic](http://pbmpb9h15.bkt.gdipper.com/ar.model.tar.xz) | [Bulgarian](http://pbmpb9h15.bkt.gdipper.com/bg.model.tar.xz) | [Catalan](http://pbmpb9h15.bkt.gdipper.com/ca.model.tar.xz) | [Czech](http://pbmpb9h15.bkt.gdipper.com/cs.model.tar.xz)  |
| [Old Church Slavonic](http://pbmpb9h15.bkt.gdipper.com/cu.model.tar.xz) | [Danish](http://pbmpb9h15.bkt.gdipper.com/da.model.tar.xz) | [German](http://pbmpb9h15.bkt.gdipper.com/de.model.tar.xz) | [Greek](http://pbmpb9h15.bkt.gdipper.com/el.model.tar.xz) | 
| [English](http://pbmpb9h15.bkt.gdipper.com/en.model.tar.xz) | [Spanish](http://pbmpb9h15.bkt.gdipper.com/es.model.tar.xz) | [Estonian](http://pbmpb9h15.bkt.gdipper.com/et.model.tar.xz) | [Basque](http://pbmpb9h15.bkt.gdipper.com/eu.model.tar.xz) |
| [Persian](http://pbmpb9h15.bkt.gdipper.com/fa.model.tar.xz) | [Finnish](http://pbmpb9h15.bkt.gdipper.com/fi.model.tar.xz) | [French](http://pbmpb9h15.bkt.gdipper.com/fr.model.tar.xz) | [Irish](http://pbmpb9h15.bkt.gdipper.com/ga.model.tar.xz) | 
| [Galician](http://pbmpb9h15.bkt.gdipper.com/gl.model.tar.xz) | [Ancient_Greek](http://pbmpb9h15.bkt.gdipper.com/grc.model.tar.xz) | [Hebrew](http://pbmpb9h15.bkt.gdipper.com/he.model.tar.xz) | [Hindi](http://pbmpb9h15.bkt.gdipper.com/hi.model.tar.xz) | 
| [Croatian](http://pbmpb9h15.bkt.gdipper.com/hr.model.tar.xz) | [Hungarian](http://pbmpb9h15.bkt.gdipper.com/hu.model.tar.xz) | [Indonesian](http://pbmpb9h15.bkt.gdipper.com/id.model.tar.xz) | [Italian](http://pbmpb9h15.bkt.gdipper.com/it.model.tar.xz) |
| [Japanese](http://pbmpb9h15.bkt.gdipper.com/ja.model.tar.xz) | [Korean](http://pbmpb9h15.bkt.gdipper.com/ko.model.tar.xz) | [Latin](http://pbmpb9h15.bkt.gdipper.com/la.model.tar.xz) | [Latvian](http://pbmpb9h15.bkt.gdipper.com/lv.model.tar.xz) |
| [Dutch](http://pbmpb9h15.bkt.gdipper.com/nl.model.tar.xz) | [Norwegian-Bokmaal](http://pbmpb9h15.bkt.gdipper.com/no_bokmaal.model.tar.xz) | [Norwegian-Nynorsk](http://pbmpb9h15.bkt.gdipper.com/no_nynorsk.model.tar.xz) | [Polish](http://pbmpb9h15.bkt.gdipper.com/pl.model.tar.xz) | 
| [Portuguese](http://pbmpb9h15.bkt.gdipper.com/pt.model.tar.xz) | [Romanian](http://pbmpb9h15.bkt.gdipper.com/ro.model.tar.xz) | [Russian](http://pbmpb9h15.bkt.gdipper.com/ru.model.tar.xz) | [Slovak](http://pbmpb9h15.bkt.gdipper.com/sk.model.tar.xz) |
| [Slovenian](http://pbmpb9h15.bkt.gdipper.com/sl.model.tar.xz) | [Swedish](http://pbmpb9h15.bkt.gdipper.com/sv.model.tar.xz) | [Turkish](http://pbmpb9h15.bkt.gdipper.com/tr.model.tar.xz) | [Uyghur](http://pbmpb9h15.bkt.gdipper.com/ug.model.tar.xz) |
| [Ukrainian](http://pbmpb9h15.bkt.gdipper.com/uk.model.tar.xz) | [Urdu](http://pbmpb9h15.bkt.gdipper.com/ur.model.tar.xz) | [Vietnamese](http://pbmpb9h15.bkt.gdipper.com/vi.model.tar.xz) | [Traditional-Chinese](http://pbmpb9h15.bkt.gdipper.com/zht.model.tar.xz) |
| [Simplified-Chinese](http://pbmpb9h15.bkt.gdipper.com/zhs.model.tar.xz) | | |


The Simplified Chinese model was trained on xinhua proportion of gigawords-v5.

## Pre-requirements

* python 3.6 
* pytorch 0.4
* other requirements from allennlp

## Usage

First, after unzip the model, please change the `"config_path"` field in `${lang}.model/config.json`
to `${project_home}/configs/cnn_50_100_512_4096_sample.json`.

Then, prepare your input file in the [conllu format](http://universaldependencies.org/format.html).
Do remember tokenization!

When it's all set, run

```
python src/gen_elmo.py test \
    --input_format conll \
    --input /path/to/your/input \
    --model /path/to/your/model \
    --output_ave /path/to/your/output
```

It will dump an hdf5 encoded `dict` onto the disk, where the key is `'\t'` separated
words in the sentence and the value is it's 3-layer averaged ELMo representation.
You can also dump the first layer using the `--output_lstm` option.
We are actively changing the interface to make it more adapted to the 
AllenNLP ELMo and more programmatically friendly.

## Training Your Own ELMo

Please run 
```
python src/biLM.py train -h
```
to get more details about the ELMo training. However, we
need to add that the training process is not very stable.
In some cases, we end up with a loss of `nan`. We are actively working on that and hopefully
improve it in the future.

## Citation

If our ELMo gave you nice improvements, please cite us.

* Wanxiang Che, Yijia Liu, Yuxuan Wang, Bo Zheng, and Ting Liu. 2018. Towards Better UD Parsing: Deep Contextualized Word Embeddings, Ensemble, and Treebank Concatenation. (to appear) In Proceedings of the CoNLL 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies (CoNLL).


## Contributor

* Bo Zheng <<bzheng@ir.hit.edu.cn>>