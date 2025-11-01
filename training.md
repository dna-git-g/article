# Training

This section describes the training regime for our models.

## Training Data and Batching

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs.  Sentences were encoded using byte-pair encoding [@DBLP:journals/corr/BritzGLL17], which has a shared source-target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [@wu2016google].  Sentence pairs were batched together by approximate sequence length.  Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.  

## Hardware and Schedule

We trained our models on one machine with 8 NVIDIA P100 GPUs.  For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds.  We trained the base models for a total of 100,000 steps or 12 hours.  For our big models,(described on the bottom line of table \ref{tab:variations}), step time was 1.0 seconds.  The big models were trained for 300,000 steps (3.5 days).

## Optimizer

We used the Adam optimizer~[@kingma2014adam] with $\beta_1=0.9$, $\beta_2=0.98$ and $\epsilon=10^{-9}$.  We varied the learning rate over the course of training, according to the formula:

\begin{equation}
lrate = d_{\text{model}}^{-0.5} \cdot
  \min\left( \text{step\_num}^{-0.5},
    \text{step\_num} \cdot \text{warmup\_steps}^{-1.5} \right)
\end{equation}

This corresponds to increasing the learning rate linearly for the first $warmup\_steps$ training steps, and decreasing it thereafter proportionally to the inverse square root of the step number.  We used $warmup\_steps=4000$.

## Regularization \label{sec:reg}

We employ three types of regularization during training:

**Residual Dropout** We apply dropout [@srivastava2014dropout] to the output of each sub-layer, before it is added to the sub-layer input and normalized.   In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.  For the base model, we use a rate of $P_{drop}=0.1$.

**Label Smoothing** During training, we employed label smoothing of value $\epsilon_{ls}=0.1$ [@DBLP:journals/corr/SzegedyVISW15].  This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.


