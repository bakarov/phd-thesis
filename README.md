# The Notion of Linguistic Representativeness of Vector Space Semantic Models 

Real-valued word representations based on co-occurrence counts have proved their efficiency in various NLP tasks. However, the problem of the most effective composition model that will help to obtain sentence representations still remains open. The most ubiquitous methods of distributional composition are based on neural networks (autoencoders, convolutional filters, etc). But these models are trained to achieve specific objectives in pre-determined tasks and do not propose a general framework. From this perspective, generalisable models able to be extended to complex semantic phenomena seem to be more representative to the lexical level of language.

Categorical Compositional Distributional (DisCoCat) framework based on categorical logic is one of those who tries to deal with the problem of construction of sentence representations. It involves mathematical operators of composition that give ability to produce representations with broad coverage and good generalisability (rather than task-specific deep learning) for complex aspects of meaning (e.g. quantifiers and their interaction). However, most of them were focused on the application of categorical framework to distributional semantic models inside only one language. This does not allow to provide fully generalisable empirical linguistic results since they would not be really language-independent. But an attempt of extension of the categorical framework to a cross-lingual setting could possibly increase its linguistic representativeness and generalizibility.

The aim of the proposed thesis is to make this attempt, providing a cross-lingual extension of the categorical framework. It could be obtained with the help of cross-language word representations that have been actively researched and developed in the NLP community in recent years. The aim of the thesis is to formulate theoretical foundations of such cross-lingual categorical framework as well as to prove its applicability through empirical experiments on cross-language word representations (like MUSE, VecMap, etc) and comparisons with existing cross-lingual distributional compositional models based on deep learning. Therefore, the possible research question could be asked as how good is the categorical framework comparing to deep learning one considering a wide range of models and languages.

## Results

### Semantic Textual Similarity

|         | STS-2016, ES-EN, mse | STS-2017, ES-EN, mse | STS-2017, TR-EN, mse |
| --------|:--------------------:| :-------------------:|:--------------------:|
| MUSE    | 2.943                | 1.979                | 2.323                |
| USE     | 0.79                 | 1.156                | 1.183                |
| SBERT   | 1.253                | 2.004                | 2.234                |


### Natural Language Inference

|         | XNLI, EN-RU, acc     |
| --------|:--------------------:| 
| MUSE    | 0.343                |
| USE     | 0.373                |
| SBERT   | 0.328                | 
