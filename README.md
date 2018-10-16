Pre-trained ELMo Representations for Many Languages
===================================================
This repo was forked from https://github.com/HIT-SCIR/ELMoForManyLangs .

It is nice of [HIT-SCIR](https://github.com/HIT-SCIR) to provide the bunch of pre-trained ELMo models. However, they only have provided "file to file" conversion from tokens to vectors with restricted formats, which is not so convenient to use.

I wrote a easy-to-use version to converting tokens to vectors. You can use it like this:

```python
#import it outside the top directory of this repo
from ELMoForManyLangs import elmo

e = elmo.Embedder()

sents = [['今', '天', '天氣', '真', '好', '阿'],
['潮水', '退', '了', '就', '知道', '誰', '沒', '穿', '褲子']]
# the list of lists which store the sentences 
# after segment if necessary.

e.sents2elmo(sents)
# will return a list of numpy arrays 
# each with the shape=(seq_len, embedding_size)
```

### the parameters of the function sents2elmo:
```python
def sents2elmo(sents, output_layer=-1):
```
- **sents**: the list of lists which store the sentences after segment if necessary.
-  **output_layer**: the target layer to output. 
    -  0 for the word encoder
    -  1 for the first LSTM hidden layer
    -  2 for the second LSTM hidden layer
    -  -1 for an average of 3 layers. (default)


## Pre-requirements

* **must** python>=3.6 (if you use python3.5, you will encounter this issue https://github.com/HIT-SCIR/ELMoForManyLangs/issues/8) 
* pytorch 0.4
* overrides
* other requirements from allennlp
* download the elmo model you need from [here](https://github.com/HIT-SCIR/ELMoForManyLangs#downloads) to the top directory.
* change line 234 of the code `elmo.py` to your new model folder name, which is `zht.model` now.
