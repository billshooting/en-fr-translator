# en-fr-translator
A En-Fr Translator trained by GPT-like architecture using 2 different methods: From Scratch & Fine-Tuning

## DataSet
- The train dataset is Wikimedia-20230407, it contains more than 1.4M lines En-Fr sentence pairs.
- The test dataset is WMT14, which is commonly used in evaluation of En-Fr translator model.

Because the repo will not contain any raw training data, before the running the codes, we should download the training dataset.

``` bash
wget https://opus.nlpl.eu/results/en&fr/corpus-result-table | gzip
```

For test dataset: WMT14, it has been preset in the module: dataset, we don't have to download it separately.


## Benchmark
The benchmark of this translation task is SacreBLEU.

## Model File
All of the trained models are too big to store on GitHub.