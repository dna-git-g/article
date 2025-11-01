# Introduction

Recurrent neural networks, long short-term memory [@hochreiter1997] and gated recurrent [@gruEval14] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [@sutskever14; @bahdanau2014neural; @cho2014learning]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [@wu2016google; @luong2015effective; @jozefowicz2016exploring].

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states $h_t$, as a function of the previous hidden state $h_{t-1}$ and the input for position $t$. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.

Recent work has achieved significant improvements in computational efficiency through factorization tricks [@Kuchaiev2017Factorization] and conditional computation [@shazeer2017outrageously], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [@bahdanau2014neural; @structuredAttentionNetworks]. In all but a few cases [@decomposableAttnModel], however, such attention mechanisms are used in conjunction with a recurrent network.

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.


