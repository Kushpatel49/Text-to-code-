# Text-to-code basic using Tenserflow

**Introduction**

Given the goal of improving software development productivity with machine learning methods, software intelligence research has attracted increasing attention in both academia and industries over the last decade. Software code intelligence techniques can help developers reduce tedious repetitive workloads, enhance the programming quality, and improve the overall software development productivity. This would reduce time spent writing software as well as reduce computational and operational costs.

[Text-to-code generation](https://paperswithcode.com/task/text-to-code-generation) is a task where we can generate code based on the natural language description. It can further be used to build an AI-powered coding assistant. Developers simply type the natural language description or the function signature to specify their intents, and the AI coding assistant can generate or complete the target function for them. This helps to accelerate implementation and also reduce their reliance on external resources.

<br>
<center><img src="https://production-media.paperswithcode.com/tasks/code_generation_0P9CcWa.png" alt="code_generation" width="600"/></center>
<br>

[CodeT5 by Salesforce](https://arxiv.org/pdf/2109.00859.pdf) is the first code-aware, encoder-decoder-based pre-trained programming language model, which enables a wide range of code intelligence applications including code understanding and generation tasks. CodeT5 achieves state-of-the-art performance on 14 sub-tasks in the CodeXGLUE code intelligence benchmark.

CodeT5 builds on an encoder-decoder framework with the same architecture as [T5 by Google](https://arxiv.org/pdf/1910.10683.pdf). The T5 model, pre-trained on [C4](https://www.tensorflow.org/datasets/catalog/c4), achieves state-of-the-art results on many NLP benchmarks while being flexible enough to be fine-tuned to a variety of important downstream tasks. T5 architecture employs denoising sequence-to-sequence (Seq2Seq) pre-training and has been shown to benefit out-of-the-box for both understanding and generation tasks in natural language.

In this notebook we will finetune CodeT5 on [MBPP - Mostly Basic Python Problems by Google Research](https://research.google/tools/datasets/mostly-basic-python-problems/) to generate code based on problem description. MBPP is a benchmark that consists of around 1,000 crowd-sourced Python programming problems, designed to be solvable by entry level programmers, covering programming fundamentals, standard library functionality, and so on. Each problem consists of a task description, code solution and 3 automated test cases.

<font color=#425066><b>The notebook demonstrates how to finetune CodeT5 on MBPP Dataset w/ TensorFlow.</b></font>

<a id="section2"><font color='#FF6F00'><h2>T5</h2></font></a>

<a id="section2a"><font color=#425066><h3>Explained</h3></font></a>

- *T5 — Text-to-Text Transfer Transformer Model* was proposed in the paper, [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683). This paper is essentially a survey of modern transfer learning techniques used in language understanding and hence proposes a unified framework that attempts to combine all language problems into a text-to-text format.

- The text-to-text framework suggests using the same model, same loss function, and the same hyperparameters on all the NLP tasks. In this approach, the inputs are modeled in such a way that the model shall recognize a task, and the output is simply the “text” version of the expected outcome.

<br>
<center><img src="https://miro.medium.com/max/1280/0*xfXDPjASztwmJlOa.gif" alt="t5" width="600"/></center>
<br>

- To avail the same model for all the downstream tasks, a task-specific text prefix is added to the original input that is fed to the model. This text prefix is also considered as a hyperparameter.
> As an example,to ask the model to translate the sentence “That is good.” from English to German, the model would be fed the sequence “translate English to German: That is good.” and would be trained to output “Das ist gut.”
— T5 Paper

 Similarly, for classification tasks, the model predicts a single word corresponding to the target label.
> For example, on the MNLI benchmark the goal is to predict whether a premise implies (“entailment”), contradicts (“contradiction”), or neither (“neutral”) a hypothesis. With our preprocessing, the input sequence becomes “mnli premise: I hate pigeons. hypothesis: My feelings towards pigeons are filled with animosity.” with the corresponding target word “entailment”.
— T5 Paper

- T5 is pretrained using the denoising objective on C4— Colossal Clean Crawled Corpus - a 750GB dataset which is not just reasonably larger than the most pre-training datasets but also contains a relatively very clean text.

<br>
<center><img src="https://miro.medium.com/max/1280/0*8WIzJDlbitZVDGeA.png" alt="t5_pretrain_finetune" width="600"/></center>
<br>

- The proposed model is essentially a Encoder-Decoder [Transformer](https://arxiv.org/abs/1706.03762) with some architectural changes (like applying Layer Normalization before a sub block and then adding the initial input to the sub-block output; also known as pre-norm). Moreover, the model configuration is similar to [BERT base](https://arxiv.org/abs/1810.04805). T5 uses relative scalar embeddings. Encoder input padding can be done on the left and on the right.

- T5 comes in different sizes: t5-small, t5-base, t5-large, t5-3b, t5-11b.

<a id="section2b"><font color=#425066><h3>Training</h3></font></a>

- T5 is an encoder-decoder model and converts all NLP problems into a text-to-text format. It is trained using teacher forcing. This means that for training, we always need an input sequence and a corresponding target sequence. The input sequence is fed to the model using `input_ids`. The target sequence is shifted to the right, i.e., prepended by a start-sequence token and fed to the decoder using the `decoder_input_ids`. In teacher-forcing style, the target sequence is then appended by the EOS token and corresponds to the `labels`. The PAD token is hereby used as the start-sequence token. T5 can be trained / fine-tuned both in a supervised and unsupervised fashion.

- One can use [T5ForConditionalGeneration](https://huggingface.co/docs/transformers/v4.18.0/en/model_doc/t5#transformers.T5ForConditionalGeneration) (or the Tensorflow/Flax variant) from [HuggingFace transformers library](https://huggingface.co/docs/transformers/), which includes the language modeling head on top of the decoder.

- T5 models need a slightly higher learning rate than the default one set in the Trainer when using the AdamW optimizer. Typically, 1e-4 and 3e-4 work well for most problems (classification, summarization, translation, question answering, question generation). Note that T5 was pre-trained using the AdaFactor optimizer.

- Task prefixes matter when (1) doing multi-task training (2) your task is similar or related to one of the supervised tasks used in T5’s pre-training mixture (see Appendix D of the [paper](https://arxiv.org/pdf/1910.10683.pdf) for the task prefixes used).

-  We must make sure that padding token id’s of the labels are not taken into account by the loss function. In PyTorch and Tensorflow, this can be done by replacing them with -100, which is the `ignore_index` of the `CrossEntropyLoss`. We also pass attention_mask as additional input to the model, which makes sure that padding tokens of the inputs are ignored.

<a id="section2c"><font color=#425066><h3>Inference</h3></font></a>

- At inference time, it is recommended to use [generate()](https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate). This method takes care of encoding the input and feeding the encoded hidden states via cross-attention layers to the decoder and auto-regressively generates the decoder output. Check out this [blog post](https://huggingface.co/blog/how-to-generate) to know all the details about generating text with Transformers. There’s also this [blog post](https://huggingface.co/blog/encoder-decoder#encoder-decoder) which explains how generation works in general in encoder-decoder models.

<a id="section2d"><font color=#425066><h3>References</h3></font></a>
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- [Google AI Blog - Exploring Transfer Learning with T5: the Text-To-Text Transfer Transformer](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)
- [HuggingFace - T5](https://huggingface.co/docs/transformers/model_doc/t5#training)
- [Understanding Transformer-Based Self-Supervised Architectures](https://towardsdatascience.com/t5-text-to-text-transfer-transformer-643f89e8905e)
- [T5 - A Detailed Explanation](https://medium.com/analytics-vidhya/t5-a-detailed-explanation-a0ac9bc53e51)


<a id="section3"><font color='#FF6F00'><h2>CodeT5</h2></font></a>

<a id="section3a"><font color=#425066><h3>Explained</h3></font></a>

-  CodeT5 by Salesforce was proposed in [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models
for Code Understanding and Generation](https://arxiv.org/pdf/2109.00859.pdf) and is an open-source model that can understand and readily generate code. It is an identifier-aware unified pre-trained coder-encoder tool that enables a wide range of code intelligence applications. CodeT5 possesses an uninformed model for natural language processing tasks, which reframes text-to-text with input, and output data always being strings of texts.


- CodeT5 builds on the similar architecture of T5 but incorporates code-specific knowledge to endow the model with better code understanding. It takes code and its accompanying comments as a sequence to build and generate upon. It aims to derive generic representations for programming language (PL) and natural language (NL) via pre-training on unlabeled source code. 

- CodeT5 achieves state-of-the-art performance on multiple code-related downstream tasks including understanding tasks such as code defect detection and clone detection, and generation tasks across various directions including PL-NL, NL-PL, and PL-PL. 


<a id="section3b"><font color=#425066><h3>Pretraining</h3></font></a>

- CodeT5 was pretrained on the [CodeSearchNet](https://github.com/github/CodeSearchNet) data that consists of both unimodal (PL-only) and bimodal (PL-NL) data on six PLs - `Ruby, JavaScript, Go, Python, PHP, C, and C#`. In addition to that, they further collect extra data of C/C# from open-source Github repositories. They further finetune CodeT5 on most tasks in the [CodeXGLUE benchmark](https://arxiv.org/pdf/2102.04664.pdf), including two understanding tasks: code defect detection and clone detection, and generation tasks such as code summarization, generation, translation, and refinement.

<br>
<center><img src="https://149695847.v2.pressablecdn.com/wp-content/uploads/2021/09/image-179-1024x406.png" alt="codet5_pretraining" width="700"/></center>
<br>

- Some of the pre-training tasks of CodeT5 include:

    - `Masked Span Prediction (MSP)` randomly masks spans with arbitrary lengths and requires the decoder to recover the original input. It captures the syntactic information of the NL-PL input and learns robust cross-lingual representations as we pre-train on multiple PLs with a shared model.
    
    - `Identifier Tagging (IT)` applied only to the encoder which distinguishes whether each code token is an identifier (e.g., variables or function names) or not. It works like the syntax highlighting feature in some developer-aided tools.
    
    - `Masked Identifier Prediction (MIP)`, in contrast to MSP, only masks identifiers and employs the same mask placeholder for all occurrences of one unique identifier. It works like deobfuscation in software engineering and is a more challenging task that requires the model to comprehend the code semantics based on the obfuscated code.
    
    - `Bimodal Dual Generation (dual-gen)` jointly optimizes the conversion from code to its comments and vice versa. It encourages a better alignment between the NL and PL counterparts. 
    
<a id="section3c"><font color=#425066><h3>Code-specific Tokenizer</h3></font></a>

- In CodeT5 they train a Byte-level BPE tokenizer and set the vocabulary size to 32,000 as T5. They add additional special tokens ([PAD], [CLS], [SEP], [MASK0], ..., [MASK99]). This tokenzier is trained on all pre-training data with non-printable characters and low-frequent tokens (occurring <3 times) filtered. 

- When compared to T5’s default tokenizer they find that the trained tokenizer largely reduces the length of tokenized code sequence by 30% - 45% on downstream tasks. This accelerates the training and especially benefits generation tasks due to the shorter sequence to predict. 

- They spot a severe problem for applying the T5’s default tokenizer on source code, where it would encode some common code tokens such as brackets [‘{’, ‘}’] into unknown tokens.

<a id="section3d"><font color=#425066><h3>References</h3></font></a>
- [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation](https://arxiv.org/pdf/2109.00859.pdf)
- [CodeT5 Blog by Salesforce](https://blog.salesforceairesearch.com/codet5/)
- [CodeT5 GitHub Repository](https://github.com/salesforce/CodeT5#fine-tuning)
- [HuggingFace CodeT5-Base Model](https://huggingface.co/Salesforce/codet5-base)
- [Salesforce’s CodeT5 system can understand and generate code](https://venturebeat.com/2021/09/07/salesforces-codet5-system-can-understand-and-generate-code/)
- [Salesforce CodeT5 vs Github Copilot: A Comparative Guide to Auto-code Generators](https://analyticsindiamag.com/salesforce-codet5-vs-github-copilot-a-comparative-guide-to-auto-code-generators/)
- [Fine_tune_CodeT5_for_generating_docstrings_from_Ruby_code](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/T5/Fine_tune_CodeT5_for_generating_docstrings_from_Ruby_code.ipynb)
- [CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation](https://arxiv.org/pdf/2102.04664.pdf)
- [CodeSearchNet](https://github.com/github/CodeSearchNet)
- [OpenAI Codex](https://openai.com/blog/openai-codex/)

<a id="section4"><font color='#FF6F00'><h2>Imports</h2></font></a>

Let's start by importing required libraries to the environment: 

- [*TensorFlow*](https://www.tensorflow.org/) an end-to-end open source platform for machine learning.
- [*transformers*](https://huggingface.co/docs/transformers/index) provides APIs to easily download and train state-of-the-art pretrained models
- [*datasets*](https://huggingface.co/docs/datasets/index) a library for easily accessing and sharing datasets.
