# Results

## Machine Translation

\input{tables/table2.tex}

On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table~\ref{tab:wmt-results}) outperforms the best previously reported models (including ensembles) by more than $2.0$ BLEU, establishing a new state-of-the-art BLEU score of $28.4$.  The configuration of this model is listed in the bottom line of Table~\ref{tab:variations}.  Training took $3.5$ days on $8$ P100 GPUs.  Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of $41.0$, outperforming all of the previously published single models, at less than $1/4$ the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate $P_{drop}=0.1$, instead of $0.3$.

For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals.  For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of $4$ and length penalty $\alpha=0.6$ \citep{wu2016google}.  These hyperparameters were chosen after experimentation on the development set.  We set the maximum output length during inference to input length + $50$, but terminate early when possible \citep{wu2016google}.

Table \ref{tab:wmt-results} summarizes our results and compares our translation quality and training costs to other model architectures from the literature.  We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU \footnote{We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively.}.
%where we compare against the leading machine translation results in the literature. Even our smaller model, with number of parameters comparable to ConvS2S, outperforms all existing single models, and achieves results close to the best ensemble model.

## Model Variations

\input{tables/table3.tex}

To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging.  We present these results in Table~\ref{tab:variations}.  

In Table~\ref{tab:variations} rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section \ref{sec:multihead}. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.

In Table~\ref{tab:variations} rows (B), we observe that reducing the attention key size $d_k$ hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting.  In row (E) we replace our sinusoidal positional encoding with learned positional embeddings \citep{JonasFaceNet2017}, and observe nearly identical results to the base model.

## English Constituency Parsing

\input{tables/table4.tex}

To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input.
Furthermore, RNN sequence-to-sequence models have not been able to attain state-of-the-art results in small-data regimes \cite{KVparse15}.

We trained a 4-layer transformer with $d_{model} = 1024$ on the Wall Street Journal (WSJ) portion of the Penn Treebank \citep{marcus1993building}, about 40K training sentences. We also trained it in a semi-supervised setting, using the larger high-confidence and BerkleyParser corpora from with approximately 17M sentences \citep{KVparse15}. We used a vocabulary of 16K tokens for the WSJ only setting and a vocabulary of 32K tokens for the semi-supervised setting.

We performed only a small number of experiments to select the dropout, both attention and residual (section~\ref{sec:reg}), learning rates and beam size on the Section 22 development set, all other parameters remained unchanged from the English-to-German base translation model. During inference, we increased the maximum output length to input length + $300$. We used a beam size of $21$ and $\alpha=0.3$ for both WSJ only and the semi-supervised setting.

Our results in Table~\ref{tab:parsing-results} show that despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar \cite{dyer-rnng:16}.

In contrast to RNN sequence-to-sequence models \citep{KVparse15}, the Transformer outperforms the BerkeleyParser \cite{petrov-EtAl:2006:ACL} even when training only on the WSJ training set of 40K sentences.


