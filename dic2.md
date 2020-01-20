Dictionary of technical words.
==============================

Table of Contents {#table-of-contents .TOCHeading}
=================

Table of Contents

[Bag of visual words (BOV) 5](#bag-of-visual-words-bov)

[Deformable part Model 6](#deformable-part-model)

[Encoder: 7](#encoder)

[Fisher vector 8](#fisher-vector)

[Sift: 8](#sift)

[Spatial Pyramid matching: 9](#spatial-pyramid-matching)

[VLFeat \-\-- fisher vector 9](#vlfeat-----fisher-vector)

[Articulated body pose estimation 10](#articulated-body-pose-estimation)

[Latent SVM -- learning structural svm with latent variables.
10](#latent-svm-learning-structural-svm-with-latent-variables.)

[Structural SVM -- structured learning
10](#structural-svm-structured-learning)

[Sequence learning: HMM or RNN or LSTM or..., Comparison
11](#sequence-learning-hmm-or-rnn-or-lstm-or-comparison)

[-Microsoft paper 13](#microsoft-paper)

[RNN 13](#rnn)

<div dir="rtl">

[مطلب آموزشی خوب 14](#مطلب-آموزشی-خوب)
</div>


[Min-char-rnn.py 14](#min-char-rnn.py)

[Sampling from distribution... 15](#sampling-from-distribution)

[RNN architectures. 16](#rnn-architectures.)

[-Multilayer RNNs 16](#multilayer-rnns)

[-2D RNN and ... 16](#d-rnn-and)

[-how to make RNNs deep? Options: 16](#how-to-make-rnns-deep-options)

[Back propagation in RNNs 17](#back-propagation-in-rnns)

[Structures and issues. 18](#structures-and-issues.)

[LSTM 19](#lstm)

<div dir="rtl">

[مثالی از پیاده سازی در تنسور فلو.
19](#مثالی-از-پیاده-سازی-در-تنسور-فلو.)
</div>


<div dir="rtl">

[راهنمای گام به گام LSTM ! 20](#راهنمای-گام-به-گام-lstm)
</div>


[Future of LSTMs! 20](#future-of-lstms)

[Multi layer LSTM 21](#multi-layer-lstm)

[Stateful recurrent model 21](#stateful-recurrent-model)

[Different ML tools. 21](#different-ml-tools.)

[GAN generative adversarial networks.
22](#gan-generative-adversarial-networks.)

[CNN 22](#cnn)

[Structures and practical notes and issues
22](#structures-and-practical-notes-and-issues)

[-Reducing network size: 22](#reducing-network-size)

[-networks that have up sampling! 22](#networks-that-have-up-sampling)

[-multi task losses. 23](#multi-task-losses.)

[-batch normalization 23](#batch-normalization)

[-dropout 23](#dropout)

[-Stochastic gradient descent 24](#stochastic-gradient-descent)

[-Resnet and gradient flow 24](#resnet-and-gradient-flow)

[Semantic segmentation. 25](#semantic-segmentation.)

[SVM SVC 26](#svm-svc)

[python tutorial 26](#python-tutorial)

[-color map f(x,y) 26](#color-map-fxy)

[debugging 26](#debugging)

[Tensorflow tutorial 26](#tensorflow-tutorial)

[-get weights and biases 27](#get-weights-and-biases)

[-autoencoder examples 28](#autoencoder-examples)

[-some notes... 28](#some-notes)

[ipython -- remote debugging 29](#ipython-remote-debugging)

[keras tutorials 29](#keras-tutorials)

[layers 30](#layers)

[-embedding 30](#embedding)

[Pytorch tutorial 31](#pytorch-tutorial)

[What is pack propagation? 31](#what-is-pack-propagation)

[Google Colab 31](#google-colab)

[Parallel programming Deep 33](#parallel-programming-deep)

[on raspberry pi 33](#on-raspberry-pi)

Dictionary of technical words. 1Speech recognition 12

Bag of visual words (BOV) 
==========================

http://what-when-how.com/computer-visionimaging-and-computer-graphics/fisher-vectors-beyond-bag-of-visual-words-image-representations-computer-visionimaging-and-computer-graphics-part-1/

The idea is very similar in computer vision too. We represent an object
as a bag of "visual words". These visual words are basically important
points in the images. These points are called "features", and they are
discriminative. What this means is that a big patch of monotonic region
is not considered to be a feature because it doesn't give us much
information. This is in contrast with a sharp corner or a unique color
combination, where we get a lot of information about the image. We can
use the BoW model for image classification by constructing a large
vocabulary of many visual words and represent each image as a histogram
of the frequency words that are in the image. To actually use BoW for
image classification, we need to extract these features, generate a
codebook, and then generate a histogram.
[[https://prateekvjoshi.com/2014/08/17/image-classification-using-bag-of-words-model/]{.underline}](https://prateekvjoshi.com/2014/08/17/image-classification-using-bag-of-words-model/)

Deformable part Model
=====================

![](.//media/image1.png){width="7.239583333333333in"
height="4.614583333333333in"}

Stanford cs231n. lec1

Encoder:
========

![](.//media/image2.png){width="4.322916666666667in"
height="4.802083333333333in"}

<div dir="rtl">

خودمان روی لایه های میانی تابع همانی محدودیت می گذاریم تا دیتا را فشرده
کند\...
</div>


it is trying to learn an approximation to the identity function, so as
to output *x*\^ that is similar to *x*.

it must try to "'reconstruct"' the 100-pixel input *x*. If the input
were completely random---say, each *xi* comes from an IID Gaussian
independent of the other features---then this compression task would be
very difficult. But if there is structure in the data, for example, if
some of the input features are correlated, then this algorithm will be
able to discover some of those correlations

-imposing other constraints on the network. In particular, if we impose
a "'sparsity"' constraint on the hidden units, then the autoencoder will
still discover interesting structure in the data, even if the number of
hidden units is large.

**Visualizing a trained autoencoder**

![](.//media/image3.png){width="4.333333333333333in" height="4.21875in"}

[[http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/]{.underline}](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)

Fisher vector
=============

While early systems were mainly used global image descriptors \[15\],
recent systems extract rather local features from image patches or
segmented image regions and use techniques based on feature matching
\[16\], build inverted files \[17\], bag-of visual words \[1\] or Fisher
Vectors \[18\]

Sift: 
======

how to match pictures, with different point of view, occlusion,
brightness,... ?

<div dir="rtl">

روشی که با منطبق کردن pattern ها در نقاط مختلف دو تصویر، آن ها را با هم
انطباق می دهد.
</div>


![](.//media/image4.png){width="7.7125in" height="3.8715277777777777in"}

Stanford cs231n. Lec1

pca
===

<div dir="rtl">

می توان جهت های مختلف را برای بیشینه شدن واریانس بررسی کرد. که C ماتریس
کوواریانس یا AAT است(اثبات شد)(به مفهوم این ماتریس هم توجه شود. واریانس
و کوواریانس در ابعاد مختلف)
</div>


<div dir="rtl">

حال برای بیشینه کردن var کدام جهت بهتر است؟ جهت بردار ویژه. (با بزرگترین
مقدار ویژه. در ادامه برای بدست آوردن بقیه جهت ها، تصویر داده ها در این
جهت را حذف می کنیم. )(چرا بردار ویژه ها متعامدند؟ چون ماتریس قرینه است)
</div>


<div dir="rtl">

چرا؟ با تصویر کردن بردار دلخواه u در جهت بردار ویژه و ماکسیمم کردن
عبارت(باتوجه بهاینکه طول بردار ثابته)، ضریب مقدار ویژه بزرگتر باید یک
بشه.
</div>


<div dir="rtl">

(مقدار ویژه منفی؟)
</div>


![](.//media/image5.png){width="4.575in" height="2.9069444444444446in"}

![](.//media/image6.png){width="6.772222222222222in"
height="2.426388888888889in"}

LDA- Fisher
===========

MLCV ta - lab3

**Spatial Pyramid matching:** 
==============================

Taking features (objects) from different parts of a picture and use all
of them to determine what is the picture(scene).

Putting all the things understood in a feature descriptor and using SVM
or ... we can determine and classify scenes.

Stanford cs231n. Lec1

representation based video recognition
======================================

Action proposals
----------------

<div dir="rtl">

با بررسی فرم یک اکشن در طول زمان، می توان خوشه بندی و طبقه بندی انجام
داد.
</div>


[[https://www.dideo.ir/v/yt/oluw16wExDY]{.underline}](https://www.dideo.ir/v/yt/oluw16wExDY)

![](.//media/image7.png){width="5.104861111111111in"
height="1.7555555555555555in"}

VLFeat \-\-- fisher vector
==========================

The **VLFeat** [[open
source]{.underline}](http://www.vlfeat.org/license.html) library
implements popular computer vision algorithms specializing in image
understanding and local features extraction and matching. Algorithms
include Fisher Vector, VLAD, SIFT, MSER, k-means, hierarchical k-means,
agglomerative information bottleneck, SLIC superpixels, quick shift
superpixels, large scale SVM training, and many others. It is written in
C for efficiency and compatibility, with interfaces in MATLAB for ease
of use, and detailed documentation throughout. It supports Windows, Mac
OS X, and Linux. Vlfeat.org

Articulated body pose estimation
================================

<div dir="rtl">

تشخیص نحوه ایستادن(موقعیت مفاصل و \...) فرد از روی تصویر.
</div>


<div dir="rtl">

راهکار ها:
</div>


A mostly used technique is the spatial structure coding, often described
by the probabilistic graphical mode

<div dir="rtl">

در مقاله زیر مروری روی مقالات انجام گرفته است.
</div>


Interactive activity recognition using pose-based spatio--temporal
relation features and four-level Pachinko Allocation Model

Latent SVM -- learning structural svm with latent variables. 
=============================================================

[[http://www.cs.cornell.edu/\~cnyu/papers/icml09\_latentssvm.pdf]{.underline}](http://www.cs.cornell.edu/~cnyu/papers/icml09_latentssvm.pdf)

Structural SVM -- structured learning 
======================================

[[http://www.robots.ox.ac.uk/\~vedaldi//assets/svm-struct-matlab/tutorial/ssvm-tutorial-handout.pdf]{.underline}](http://www.robots.ox.ac.uk/~vedaldi//assets/svm-struct-matlab/tutorial/ssvm-tutorial-handout.pdf)

<div dir="rtl">

این خوبه. \...
</div>


Sequence learning: HMM or RNN or LSTM or..., Comparison
=======================================================

LSTM can learn to recognize [[context-sensitive
languages]{.underline}](https://en.wikipedia.org/wiki/Context-sensitive_languages)
unlike previous models based on [[hidden Markov
models]{.underline}](https://en.wikipedia.org/wiki/Hidden_Markov_model)
(HMM) and similar
concepts.[^[\[42\]]{.underline}^](https://en.wikipedia.org/wiki/Recurrent_neural_network#cite_note-42)
[[https://en.wikipedia.org/wiki/Recurrent\_neural\_network]{.underline}](https://en.wikipedia.org/wiki/Recurrent_neural_network)

Hierarchical RNNs connect their neurons in various ways to decompose
hierarchical behavior into useful
subprograms.^[[\[33\]](https://en.wikipedia.org/wiki/Recurrent_neural_network#cite_note-schmidhuber1992-33)[\[52\]](https://en.wikipedia.org/wiki/Recurrent_neural_network#cite_note-52)]{.underline}^

![](.//media/image8.png){width="7.7125in" height="3.7444444444444445in"}

Sentiment classification: seq of words-\> sentiment(whether + of -)

Machine translation.

Video classification on frame level.

Seq processing of non seq data.

**Searching for interpretable cells:**

<div dir="rtl">

برای اینکه معنای hidden states را پیدا کند، مقدار هر بعد اسکالر را روی
متن نشان داد. بعضی شان معنای آشنایی برای ما دارند\...
</div>


<div dir="rtl">

مقاله ای از Justin Johnson
</div>


![](.//media/image9.png){width="5.920138888888889in"
height="3.328472222222222in"}

Speech recognition
------------------

[[https://www.svds.com/tensorflow-rnn-tutorial/]{.underline}](https://www.svds.com/tensorflow-rnn-tutorial/):

Training the acoustic model for a traditional speech recognition
<div dir="rtl">

pipeline that uses Hidden Markov Models (HMM) requires speech+text data,
as well as a word to phoneme dictionary. HMMs are generative
</div>

probabilistic models for sequential data, and are typically evaluated
using [[Levenshtein word error
distance]{.underline}](https://en.wikipedia.org/wiki/Levenshtein_distance),
a string metric for measuring differences in strings. These models can
be simplified and made more accurate with speech data that is aligned
with phoneme transcriptions, but this a tedious manual task.

RNNs use **Connectionist Temporal Classification (CTC) loss
function...**

We can discard the concept of phonemes when using neural networks for
speech recognition by using an objective function that allows for the
prediction of character-level transcriptions: [[Connectionist Temporal
Classification]{.underline}](http://www.cs.toronto.edu/~graves/icml_2006.pdf)
(CTC). Briefly, CTC enables the computation of probabilities of multiple
sequences, where the sequences are the set of all possible
character-level transcriptions of the speech sample

If you want to learn more about CTC, there are many papers and [[blog
posts]{.underline}](https://gab41.lab41.org/speech-recognition-you-down-with-ctc-8d3b558943f0?gi=24cd18fe52c#.wbsc6x23a)
that explain it in more detail. We will use TensorFlow's [[CTC
implementation]{.underline}](https://www.tensorflow.org/api_guides/python/nn#Connectionist_Temporal_Classification_CTC_),
and there continues to be research and improvements on CTC-related
implementations, such as [[this recent
paper]{.underline}](https://arxiv.org/abs/1703.00096) from Baidu.

-Microsoft paper
----------------

In September 2016, Microsoft released a [[paper in
arXiv]{.underline}](https://arxiv.org/abs/1609.03528) describing how
they achieved a 6.9% error rate on the NIST 200 Switchboard data. They
utilized several different acoustic and language models on top of their
convolutional+recurrent neural network. Several key improvements that
have been made by the Microsoft team and other researchers in the past 4
years include:

-   using language models on top of character based RNNs

-   using convolutional neural nets (CNNs) for extracting features from
    the audio

-   ensemble models that utilize multiple RNNs

It is important to note that the language models that were pioneered in
traditional speech recognition models of the past few decades, are again
proving valuable in the deep learning speech recognition models.

RNN
===

A finite impulse recurrent network is a [[directed acyclic
graph]{.underline}](https://en.wikipedia.org/wiki/Directed_acyclic_graph)
that can be unrolled and replaced with a strictly feedforward neural
network, while an infinite impulse recurrent network is a [[directed
cyclic
graph]{.underline}](https://en.wikipedia.org/wiki/Directed_cyclic_graph)
that can not be unrolled.
[[https://en.wikipedia.org/wiki/Recurrent\_neural\_network]{.underline}](https://en.wikipedia.org/wiki/Recurrent_neural_network)

<div dir="rtl">

آیا interstate transition در RNN ها ایجاد می کنند؟ که شبیه مارکوف شود؟
</div>


![](.//media/image10.png){width="5.4625in" height="2.576388888888889in"}

<div dir="rtl">

فک کنم هر چه زمان جلو می رود، Xt-1 ها دور ریخته می شود، و اگر شکل بالا
را در نظر بگیریم، کاملترین state، مربوط به سومین بلوک است. که در زمان
بعدی، آن را به بلوک بعدی می دهد\...
</div>


<div dir="rtl">

مطلب آموزشی خوب
</div>

---------------

[[http://www.deeplearningbook.org/contents/rnn.html]{.underline}](http://www.deeplearningbook.org/contents/rnn.html)

Min-char-rnn.py
---------------

<div dir="rtl">

کدی که cs231n پیشنهاد کرد\...
</div>


Sampling from distribution...
-----------------------------

![](.//media/image11.png){width="7.7125in" height="4.336111111111111in"}

RNN architectures. 
-------------------

-Multilayer RNNs
----------------

![](.//media/image12.png){width="7.7125in" height="4.336111111111111in"}

-2D RNN and ...
---------------

Sometimes it refers only to the position in the sequence. RNNs may also
be applied in two dimensions across spatial data such as images, and
even when applied to data involving time,the network may have
connections that go backward in time, provided that the entire sequence
is observed before it is provided to the network

-how to make RNNs deep? Options:
--------------------------------

Feedforward depth. Recurrent depth. \[le.proc.9.s12
azebni.cs.illinois.edu/spring17/lec20\_rnn.pdf\]

Back propagation in RNNs
------------------------

![](.//media/image13.png){width="6.323611111111111in"
height="3.3270833333333334in"}

\[le.proc.9.s12 azebni.cs.illinois.edu/spring17/lec20\_rnn.pdf\]

![](.//media/image14.png){width="6.830555555555556in"
height="3.5388888888888888in"}

![](.//media/image15.png){width="5.304861111111111in"
height="2.6347222222222224in"}

Structures and issues. 
-----------------------

\- np-RNN

![](.//media/image16.png){width="4.577083333333333in"
height="2.254166666666667in"}initialize weight matrix to ...

\[le.proc.9.s37 azebni.cs.illinois.edu/spring17/lec20\_rnn.pdf\]

LSTM
====

LSTM works even given long delays between significant events and can
handle signals that mix low and high frequency components.

It's good to know that TensorFlow provides APIs for Python, C++,
Haskell, Java, Go, Rust, and there's also a third-party package for R
called tensorflow.

![](.//media/image17.png){width="5.422916666666667in"
height="2.615972222222222in"}

Imagine gates as binary values. / cell states inc/dec by 1. Forget or
remember them.

\* Element-wise multiplication is much better than full matrix
multiplication in RNN.

<div dir="rtl">

مثالی از پیاده سازی در تنسور فلو. 
</div>

----------------------------------

[[https://www.svds.com/tensorflow-rnn-tutorial/]{.underline}](https://www.svds.com/tensorflow-rnn-tutorial/)

<div dir="rtl">

مطالب کلی خوبی گفته، ولی راهنمای خوبی نیست. البته کد را در گیت هاب
گذاشته \...
</div>


[*[https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537]{.underline}*](https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537)

<div dir="rtl">

[مثال خیلی
خوبیه](https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537).
به نظر میرسه این شبکه POS کلمات را می فهمد\... چگونه؟ چرا نود های مخفی
512 تا است؟
</div>


<div dir="rtl">

راهنمای گام به گام LSTM !
</div>

-------------------------

[[http://colah.github.io/posts/2015-08-Understanding-LSTMs/]{.underline}](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

Future of LSTMs!
----------------

LSTMs were a big step in what we can accomplish with RNNs. It's natural
to wonder: is there another big step? A common opinion among researchers
is: "Yes! There is a next step and it's attention!" The idea is to let
every step of an RNN pick information to look at from some larger
collection of information. For example, if you are using an RNN to
create a caption describing an image, it might pick a part of the image
to look at for every word it outputs. In fact, [[Xu, *et
al.*]{.underline}
[(2015)]{.underline}](http://arxiv.org/pdf/1502.03044v2.pdf) do exactly
this -- it might be a fun starting point if you want to explore
attention! There's been a number of really exciting results using
attention, and it seems like a lot more are around the corner...

Attention isn't the only exciting thread in RNN research. For example,
Grid LSTMs by [[Kalchbrenner, *et al.*]{.underline}
[(2015)]{.underline}](http://arxiv.org/pdf/1507.01526v1.pdf) seem
extremely promising. Work using RNNs in generative models -- such as
[[Gregor, *et al.*]{.underline}
[(2015)]{.underline}](http://arxiv.org/pdf/1502.04623.pdf), [[Chung, *et
al.*]{.underline}
[(2015)]{.underline}](http://arxiv.org/pdf/1506.02216v3.pdf), or [[Bayer
& Osendorfer (2015)]{.underline}](http://arxiv.org/pdf/1411.7610v3.pdf)
-- also seems very interesting. The last few years have been an exciting
time for recurrent neural networks, and the coming ones promise to only
be more so!

Multi layer LSTM
----------------

![](.//media/image18.png){width="6.011111111111111in"
height="3.859027777777778in"}

Stateful recurrent model
------------------------

A stateful recurrent model is one for which the internal states
(memories) obtained after processing a batch of samples are reused as
initial states for the samples of the next batch. This allows to process
longer sequences while keeping computational complexity manageable.

Search algorithms
=================

![](.//media/image19.png){width="6.315972222222222in"
height="3.438888888888889in"}

we wanna maximize this conditional probability. Not to choose words
randomly.

![](.//media/image20.png){width="3.877083333333333in" height="0.8756944444444444in"}Greedy search:
--------------------------------------------------------------------------------------------------

![](.//media/image21.png){width="5.377083333333333in"
height="2.99375in"} choose words one by one according to previous words.
But not a complete sentence is generated at first step.

Beam search: 
-------------

(to deal with large space of decoding) instead of considering one word
as most probable, consider eg 3 words and compute prob over them(3 =
beam width). At last consider most probable triplet instead of just
independent words.

At step 2, again choose the most 3 probable pairs from a 30000 choice
space.

<div dir="rtl">

پس به جای اینکه سافتمکس بعد انکودر کلمه اول، یا کل جمله(که هرکدام
محدودیت هایی ایجاد می کند) را انتخاب کند، در هر استپ سه گزینه را نگه می
دارد. به علاوه هم چنان سافتمکس یک کلمه را انتخاب می کند.
</div>


Different ML tools.
===================

-   [[Apache MXNet
    Gluon]{.underline}](https://mxnet.incubator.apache.org/api/python/gluon/data.html)

-   [[deeplearn.js]{.underline}](https://deeplearnjs.org/demos/model-builder/)

-   [[Kaggle]{.underline}](https://www.kaggle.com/zalando-research/fashionmnist)

-   [[Pytorch]{.underline}](http://pytorch.org/docs/master/torchvision/datasets.html#fashion-mnist)

-   [[Keras]{.underline}](https://keras.io/datasets/#fashion-mnist-database-of-fashion-articles)

-   [[Edward]{.underline}](http://edwardlib.org/api/observations/fashion_mnist)

-   [[Tensorflow]{.underline}](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist)

-   [[Torch]{.underline}](https://github.com/mingloo/fashion-mnist)

-   [[JuliaML]{.underline}](https://github.com/JuliaML/MLDatasets.jl)

-   [[Chainer]{.underline}](https://docs.chainer.org/en/stable/reference/generated/chainer.datasets.get_fashion_mnist.html)

GAN generative adversarial networks.
====================================

[[https://github.com/hwalsuklee/tensorflow-generative-model-collections]{.underline}](https://github.com/hwalsuklee/tensorflow-generative-model-collections)

: some models of GAN.

CNN
===

Structures and practical notes and issues
-----------------------------------------

-Reducing network size:
-----------------------

Global average pooling. ...

Googlenet and resnet are smaller than alexnet, with higher accuracy.

-networks that have up sampling!
--------------------------------

<div dir="rtl">

در ابتدا یک تابع مشخص داریم که upsample می کند، و سپس یک عملیات
کانولوشنی(transpose convolution) که ضرایبش را یاد می گیریم.
</div>


![](.//media/image22.png){width="6.302083333333333in"
height="3.5430555555555556in"}

![](.//media/image23.png){width="6.302083333333333in"
height="3.5430555555555556in"}

-multi task losses. 
--------------------

<div dir="rtl">

مثلا شبکه های که قراره هم لیبل عکس و هم موقعیت شیء در عکس را مشخص کند.
</div>


<div dir="rtl">

Loss هایی با واحد های متفاوت، آیا باعث نمی شود که در گرادینت، یکی شان
غالب شود؟
</div>


<div dir="rtl">

خب additional hyper parameter for weighting these losses به کار می بریم.
ولی مشکلی که پیش می آید، اینست که برای به دست آوردن مقدار بهینه این
پارامتر، نمی توان total losses را با هم مقایسه کرد. (چون اکنون فرمولشان
با هم تفاوت می کند.) اینجا از سنجه های دیگری برای بررسی کارآیی استفاده
می کنند. (cs231n lec11)
</div>


-batch normalization
--------------------

-dropout
--------

To prevent overfitting.

-Stochastic gradient descent
----------------------------

To economize on the computational cost at every iteration, stochastic
gradient descent
[[samples]{.underline}](https://en.wikipedia.org/wiki/Sampling_(statistics))
a subs![](.//media/image24.png){width="2.4375in"
height="2.6770833333333335in"}et of summand functions at every step.
This is very effective in the case of large-scale machine learning
problems

-Resnet and gradient flow
-------------------------

![](.//media/image25.png){width="3.09375in"
height="3.4166666666666665in"}

Lec10 cs231n.

<div dir="rtl">

میگه نسبتاً ساده می‌تواند یاد بگیرد که از برخی لایه ها استفاده نکند و با
صفر کردن وزن ها، تنها identity باشد.
</div>


<div dir="rtl">

بعلاوه regularization هم منطقی‌تر است. یعنی وقتی ضرایب را به صفر شدن هل
بدهیم در اینجا یعنی از لایه‌ای استفاده نکنیم...
</div>


<div dir="rtl">

همچنین این مسیر همانی، یک بزرگراه برای gradient flow ایجاد می کند. یعنی
گرادیان بدون تغییر به لایه‌های قبلی تأثیر می گذارد.
</div>


--

![](.//media/image26.png){width="7.7125in" height="3.776388888888889in"}

Lec10 cs231n

<div dir="rtl">

می گوید مسیرهای مستقیم که اخیرا در شبکه های جدید استفاده می شوند، کمک می
کنند تا راحتتر گرادیان ها محاسبه شوند.
</div>


Semantic segmentation.
======================

![](.//media/image27.png){width="7.7125in" height="3.152083333333333in"}

SVM SVC
=======

<div dir="rtl">

SVM هم می تواند در طبقه بندی عکس خوب باشد! لینک زیر اثر SVC های مختلف را
برای دیتاست fashion MNIST نشان می دهد:
</div>


[[http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/\#]{.underline}](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/)

<div dir="rtl">

شبکه‌های attention-based
</div>

========================

[[http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/]{.underline}](http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

python tutorial
===============

-color map f(x,y)
-----------------

\#xx,yy = np.mgrid\[0:3:30j, 0:2:30j\]

\#f = np.exp(-x\*\*2)\*np.sin(y)

\#plt.contourf(f)

\#plt.contourf(f,levels=100)

debugging
---------

<div dir="rtl">

خیلی به درد بخوره! Ctrl F5 . متغیرهایی هم که درون تابع تعریف شده است را
می شود دید. (البته اگر break point داخل تابع باشد!)
</div>


Tensorflow tutorial
===================

https://www.datacamp.com/community/tutorials/tensorflow-tutorial

Tf.constant .variable .placeholder

\# Import \`tensorflow\`\
import tensorflow as tf\
\
\# Initialize two constants\
x1 = [[tf.constant]{.underline}](http://tf.constant/)(\[1,2,3,4\])\
x2 = [[tf.constant]{.underline}](http://tf.constant/)(\[5,6,7,8\])\
\
\# Multiply\
result = [[tf.multiply]{.underline}](http://tf.multiply/)(x1, x2)\
\
\# Intialize the Session\
sess = [[tf.Session]{.underline}](http://tf.Session/)()\
\
\# Print the result\
print(sess.run(result))\
\
\# Close the session\
[[sess.close]{.underline}](http://sess.close/)()

interactive session:

\#init session and run result

With tf.session() as sess:

Output = sess.run(result)

Print(output)

You can pass the *config* and *ConfigProto*

Config=tf.ConfigProto(log\_device\_placement=True)

<div dir="rtl">

برای اینکه متغیرها مقداردهی شوند:
</div>


w\_1 = tf.Variable(\[1,2,3\])

sess.run(tf.global\_variables\_initializer())

<div dir="rtl">

محاسبه **گرادیان** ها در backprop {#محاسبه-گرادیان-ها-در-backprop .ListParagraph}
</div>

---------------------------------

<div dir="rtl">

فک کنم اینکار را فقط برای tf.variable می کند.
</div>


w\_1 = tf.Variable(\[1.0,2,3\])

w\_2 = tf.Variable(ar(\[3,4,5.0\]),dtype=\'float32\')

z\_1 = tf.multiply(w\_1,w\_2)

z\_2 = tf.multiply(z\_1,z\_1)

step =
tf.train.GradientDescentOptimizer(0.01).compute\_gradients(z\_2,var\_list=\[w\_1\])

sess = tf.Session()

sess.run(tf.global\_variables\_initializer())

print(sess.run(step))

sess.close()

\[(array(\[ 18., 64., 150.\], dtype=float32), array(\[1., 2., 3.\],
dtype=float32))\]

<div dir="rtl">

این تابع خیلی جالبه!
</div>


def load\_data(data\_directory):

directories = \[d for d in os.listdir(data\_directory)

if os.path.isdir(os.path.join(data\_directory, d))\]

labels = \[\]

images = \[\]

for d in directories:

label\_directory = os.path.join(data\_directory, d)

file\_names = \[os.path.join(label\_directory, f)

for f in os.listdir(label\_directory)

if f.endswith(\".ppm\")\]

for f in file\_names:

images.append(skimage.data.imread(f))

labels.append(int(d))

return images, labels

ROOT\_PATH = \"/your/root/path\"

train\_data\_directory = os.path.join(ROOT\_PATH,
\"TrafficSigns/Training\")

test\_data\_directory = os.path.join(ROOT\_PATH,
\"TrafficSigns/Testing\")

images, labels = load\_data(train\_data\_directory)

<div dir="rtl">

این هم کد خوبیه!
</div>


[[https://github.com/RaghavPrabhu/Deep-Learning/blob/master/digit\_recogniser/MNIST\_TF\_v1.ipynb]{.underline}](https://github.com/RaghavPrabhu/Deep-Learning/blob/master/digit_recogniser/MNIST_TF_v1.ipynb)

<div dir="rtl">

مثال از کد بالا: /home/amir/Desktop/arefi/vision97/hw3-tf
</div>

classif/trclassification.py

-get weights and biases
-----------------------

[[https://stackoverflow.com/questions/47660917/tensorflow-retrieving-weights-biases-of-the-trained-feedforward-neural-network]{.underline}](https://stackoverflow.com/questions/47660917/tensorflow-retrieving-weights-biases-of-the-trained-feedforward-neural-network)

[[https://github.com/google/prettytensor/issues/6]{.underline}](https://github.com/google/prettytensor/issues/6)

-autoencoder examples
---------------------

[[https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3\_NeuralNetworks/autoencoder.py]{.underline}](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py)

-some notes...
--------------

<div dir="rtl">

در تنسورفلو لایه‌های شبکه را به صورت مرحله به مرحله و توسط عملیات های
اولیه مثل ضرب ماتریسی و ... می سازیم. تنسورفلو شکل شبکه را ذخیره می‌کند
و از این پس اگر ورودی بدهیم خروجی را حساب می‌کند و همچنین در
backpropagation می‌تواند از روی ساختار شبکه و ورودی و خروجی مورد انتظار
ضرایب را اصلاح کند. ضرایبی که به صورت tf.varaible هستند در حین train
تغییر می‌کنند.
</div>


<div dir="rtl">

هم چنین در ابتدای تعریف شبکه، خودمان مقدار اولیه آی به این متغیرها می
دهیم:
</div>


\#a filter / kernel tensor of shape \[filter\_height, filter\_width,
in\_channels, out\_channels\],

conv1\_w = tf.Variable(tf.truncated\_normal(shape = \[5,5,1,6\], mean =
0, stddev = 0.1))

conv1\_b = tf.Variable(tf.zeros(6))

conv1 = tf.nn.conv2d(x,conv1\_w, strides = \[1,1,1,1\], padding =
\'VALID\') + conv1\_b

conv2d:

\# Given an input tensor of shape \[batch, in\_height, in\_width,
in\_channels\] and a filter / kernel tensor of shape \[filter\_height,
filter\_width, in\_channels, out\_channels\],

<div dir="rtl">

ورودی و خروجی شبکه را در هنگام تعریف شبکه نمی دهیم. تنها از
tf.placeholder استفاده می‌کنیم و بعداً در train ورودی را مقدار می دهیم.
</div>


<div dir="rtl">

**افزودن رگولاریزیشن**
</div>


reg\_losses = tf.get\_collection(tf.GraphKeys.REGULARIZATION\_LOSSES)

reg\_constant = 0.01 \# Choose an appropriate one.

loss = my\_normal\_loss + reg\_constant \* sum(reg\_losses)

\# overfitting: high epoch number. regularization. changing optimizer(in
relation to epoch num). drop out. ...

ipython -- remote debugging
---------------------------

<div dir="rtl">

می‌توان در اسپایدر به صورت ریموت به یک ipython وصل شد. ولی یک راه هم این
است که به کمک دستورات ipython ماژول را اجرا کرده و مثل اسپایدر به تک تک
متغیرها دسترسی پیدا کرد. خیلی با اسپایدر فرقی ندارد. (مگر احتمالاً در
نمایش عکس)
</div>


[[http://pages.physics.cornell.edu/\~myers/teaching/ComputationalMethods/python/ipython.html]{.underline}](http://pages.physics.cornell.edu/~myers/teaching/ComputationalMethods/python/ipython.html)

<div dir="rtl">

خیلی خوبه!
</div>


Checkpoint manager -- gradient tape
-----------------------------------

<div dir="rtl">

دو مفهوم کاربردی در تنسور فلو.
</div>


GradientTape

Computes the gradient using operations recorded in context of this tape.

Checkpoint manager

![](.//media/image28.png){width="7.7125in" height="5.001388888888889in"}

keras tutorials
===============

[[https://samyzaf.com/ML/cifar10/cifar10.html]{.underline}](https://samyzaf.com/ML/cifar10/cifar10.html)
good but out of date

\* data generation.

<div dir="rtl">

[[https://keras.io/]{.underline}](https://keras.io/) یک توتوریال مختصر و
مفید!
</div>


[[https://keras.io/getting-started/sequential-model-guide/]{.underline}](https://keras.io/getting-started/sequential-model-guide/)
also has lstm\*\*

In the [[examples
folder]{.underline}](https://github.com/keras-team/keras/tree/master/examples),
you will also find example models for real datasets:

-   CIFAR10 small images classification: Convolutional Neural Network
    (CNN) with realtime data augmentation

-   IMDB movie review sentiment classification: LSTM over sequences of
    words

-   Reuters newswires topic classification: Multilayer Perceptron (MLP)

-   MNIST handwritten digits classification: MLP & CNN

-   Character-level text generation with LSTM

layers
------

-embedding
----------

[[https://keras.io/layers/embeddings/]{.underline}](https://keras.io/layers/embeddings/)

Instead, in an embedding, words are represented by dense vectors where a
vector represents the projection of the word into a continuous vector
space.

The position of a word within the vector space is learned from text and
is based on the words that surround the word when it is used.

The position of a word in the learned vector space is referred to as its
embedding.

Two popular examples of methods of learning word embeddings from text
include:

-   Word2Vec.

-   GloVe.

Pytorch tutorial
================

What is pack propagation?
-------------------------

<div dir="rtl">

هر تابعی که برای شبکه acyclic graph حساب می شود، به صورت f(g(h(x))) می
باشد.(حتی اگر جمع و ضرب و \... داشته باشد به همین فرم در می آید.)
قابلیتی که پای تورچ ایجاد می کند اینست که وقتی مثلا تابع g محاسبه می
شود، مشتق(نسبت به x) را به کمک مقدار تابع در آن نقطه حساب می کند. حال
برای محاسبه مشتق نسبت به x می توان از مشتق هایی که حساب کرده‌ایم یعنی
f'(x) و g'(x) استفاده کرد. فقط در مرحله آخر باید دقت کرد که xهای تابع
در‌ واقع ضرایب هستند نه مقدار نود ها!(و مشتق در مقدار نودها ضرب می شود)
</div>


Google Colab
============

[[https://www.dideo.ir/v/yt/inN8seMm7UI]{.underline}](https://www.dideo.ir/v/yt/inN8seMm7UI)

[[https://medium.com/tensorflow/colab-an-easy-way-to-learn-and-use-tensorflow-d74d1686e309]{.underline}](https://medium.com/tensorflow/colab-an-easy-way-to-learn-and-use-tensorflow-d74d1686e309)

<div dir="rtl">

یک سایت آنلاین برای ران کردن کدهای پایتون . دارای پکیج های کاربردی
مختلف.
</div>


<div dir="rtl">

کافیست sign in کنید:
</div>


[[https://colab.research.google.com/notebooks/welcome.ipynb]{.underline}](https://colab.research.google.com/notebooks/welcome.ipynb)

![](.//media/image29.png){width="7.7125in"
height="2.5388888888888888in"}

<div dir="rtl">

تنسور فلو و دیگر پکیج ها را دارد. طبق گفته دانشجویان سرعت خوبی هم
دارد... کد ها و متن ها در فرمت jypter notebooks هست.
</div>


research.google.com/seedbank

<div dir="rtl">

این سایت هم دارای توتوریال هایی به صورت همین notebook ها است.
</div>


<div dir="rtl">

اجرای فایل .py
</div>


![](.//media/image30.png){width="7.7125in" height="4.770833333333333in"}

<div dir="rtl">

برای upload کردن فولدر ابتدا آن را zip کنید و سپس با دستور زیپ unzip
کنید:
</div>


!unzip Vnect-master.zip

Parallel programming Deep
=========================

on raspberry pi
---------------

[[https://github.com/nineties/py-videocore]{.underline}](https://github.com/nineties/py-videocore)

standards for coding
====================

<div dir="rtl">

وقتی قرار است پردازش طولانی روی مثلا یک دیتاست انجام شود، خروجی را چند
iteration یکبار ذخیره کنید. چون ممکن است خطای ناگهانی رخ دهد. مثلا gpu
ارور بدهد!
</div>


Programming paradigms
=====================

[[https://en.wikipedia.org/wiki/Programming\_paradigm]{.underline}](https://en.wikipedia.org/wiki/Programming_paradigm)

Diff object oriented and procedural

[[procedural]{.underline}](https://en.wikipedia.org/wiki/Procedural_programming)
which groups instructions into procedures,

[[object-oriented]{.underline}](https://en.wikipedia.org/wiki/Object-oriented_programming)
which groups instructions together with the part of the state they
operate on.

In these languages,
[[data]{.underline}](https://en.wikipedia.org/wiki/Data) and methods to
manipulate it are kept as one unit called an
[[object]{.underline}](https://en.wikipedia.org/wiki/Object_(computer_science)).
With perfect
[[encapsulation]{.underline}](https://en.wikipedia.org/wiki/Encapsulation_(computer_programming)),
one of the distinguishing features of OOP, the only way that another
object or user would be able to access the data is via the object\'s
[*[methods]{.underline}*](https://en.wikipedia.org/wiki/Method_(computer_programming)).
Thus, the inner workings of an object may be changed without affecting
any code that uses the object.

Licenses
========

[[https://choosealicense.com/licenses/]{.underline}](https://choosealicense.com/licenses/)

good brief explanation of some licenses

[![](.//media/image31.png){width="7.7125in"
height="1.7909722222222222in"}](https://tldrlegal.com/license/mit-license)[https://tldrlegal.com/license/mit-license]{.underline}

![](.//media/image32.png){width="7.7125in" height="2.877083333333333in"}

<div dir="rtl">

ابزارهای مرور مقالات و جستجو
</div>

============================

<div dir="rtl">

گوگل
</div>


swmantic scholar

arxiv

<div dir="rtl">

مقالات citing , reference را لیست می کند
</div>


<div dir="rtl">

سمانتیک اسکولار بهترینه.
</div>


<div dir="rtl">

Related Papers هم داره
</div>


datasets
========

Activity net captions

[[http://activity-net.org/challenges/2016/download.html\#c3d]{.underline}](http://activity-net.org/challenges/2016/download.html#c3d)

[[https://cs.stanford.edu/people/ranjaykrishna/densevid/]{.underline}](https://cs.stanford.edu/people/ranjaykrishna/densevid/)

[[http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/files/activity\_net.v1-3.min.json]{.underline}](http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json)

[[http://activity-net.org/download.html]{.underline}](http://activity-net.org/download.html)

[[http://activity-net.org/challenges/2018/tasks/guest\_kinetics.html]{.underline}](http://activity-net.org/challenges/2018/tasks/guest_kinetics.html)
\*\*\*

also used for action localization

some other

![](.//media/image33.png){width="4.541666666666667in"
height="0.7916666666666666in"}

msrvtt download videos.

Raspberry pi -clone and tf
==========================

Clone- **create** an image... {#clone--create-an-image .ListParagraph}
-----------------------------

[[https://raspberrypi.stackexchange.com/questions/66824/can-i-copy-paste-the-sd-card-and-use-all-developed-programs]{.underline}](https://raspberrypi.stackexchange.com/questions/66824/can-i-copy-paste-the-sd-card-and-use-all-developed-programs)

piclone or df -h
([[https://magpi.raspberrypi.org/articles/back-up-raspberry-pi]{.underline}](https://magpi.raspberrypi.org/articles/back-up-raspberry-pi))

<div dir="rtl">

فراموش نشود که درایو مثلا ۷-۸گیگی و درایو boot باید umount شود.
</div>


<div dir="rtl">

توجه شود که در هر دو دستور بالا و پایین bs=4M باشد(هنوز تست نشده)
</div>


<div dir="rtl">

چون از روی sd card هشت گیگی image گرفته و روی ۱۶ گیگی ریخته ایم، به نظر
میرسد از همه حافظه استفاده نشده(با دستور lsblk -p) احتمالا نیاز به
پارتیشن بندی مجدد دارد
</div>


<div dir="rtl">

کارت حافظه rpi3B+ روی rpi3B کار می کند!(تنسورفلو هم دارد\...
</div>


[[https://www.raspberrypi.org/documentation/installation/installing-images/linux.md]{.underline}](https://www.raspberrypi.org/documentation/installation/installing-images/linux.md)

<div dir="rtl">

یه مشکلی که بود : پس از اینکه بچه ها گرفتند، بعضی sd ها کار نمی کرد. و
کلا umount هم نمیشد. مجبور شدمformat کنم و درست شد. هر دو boot ,... را.
</div>


<div dir="rtl">

**اتصال به اینترنت با موبایل-سریع**
</div>


<div dir="rtl">

یکی از سریعترین راه ها برای اتصال برد به اینترنت استفاده از usb
tethering است. این قابلیت در موبایل های نسبتا جدید هست. پس از اتصال
موبایل با کابل usb به برد، می توان در تنظیمات connection گزینه usb
tethering را فعال کرد. و با اتصال موبایل به اینترنت دیتا، برد نیز
اینترنت می گیرد.
</div>


<div dir="rtl">

**مشکل اتصال برد ها به wifi**
</div>


<div dir="rtl">

احتمال دارد بردهای آزمایشگاه، چیپ wifi متفاوتی داشته باشند. چون image ها
روی بعضی از برد ها به اینترنت وصل نمی شود.
</div>


<div dir="rtl">

**نصب سریع پکیچ ها با wheel**
</div>


<div dir="rtl">

piwheel.org/simple/pandas یا آدرس های مشابه
</div>


<div dir="rtl">

از این آدرس فایل whl مورد نظر را دانلود کرده و pip install filename.whl
را وارد کنید. در env هم می شود. من armv7l را دانلود کردم. هرچند arm8
است.
</div>


install tf
----------

[[https://maker.pro/raspberry-pi/tutorial/how-to-set-up-the-machine-learning-software-tensorflow-on-raspberry-pi]{.underline}](https://maker.pro/raspberry-pi/tutorial/how-to-set-up-the-machine-learning-software-tensorflow-on-raspberry-pi)

[[https://magpi.raspberrypi.org/articles/tensorflow-ai-raspberry-pi]{.underline}](https://magpi.raspberrypi.org/articles/tensorflow-ai-raspberry-pi)

<div dir="rtl">

روی بردهای قدیمی جواب نداد و دارم از این استفاده می کنم
</div>

[[https://www.tensorflow.org/install/source\_rpi]{.underline}](https://www.tensorflow.org/install/source_rpi)

<div dir="rtl">

نصب داکر برای خط بالا:
</div>

[[https://dev.to/rohansawant/installing-docker-and-docker-compose-on-the-raspberry-pi-in-5-simple-steps-3mgl]{.underline}](https://dev.to/rohansawant/installing-docker-and-docker-compose-on-the-raspberry-pi-in-5-simple-steps-3mgl)

<div dir="rtl">

این هم نشد! مجبورم ورژن پایتون را 3.4\< بکنم چون باهاش کار نمی کنه.
</div>


[[https://gist.github.com/dschep/24aa61672a2092246eaca2824400d37f]{.underline}](https://gist.github.com/dschep/24aa61672a2092246eaca2824400d37f)

<div dir="rtl">

اگر وایفای رسپبری کار نکرد، در تنظیمات وایفای کشورش را به ایران تغییر
دهید
</div>


vnc
---

<div dir="rtl">

از vncviewer که با همین دستور در ترمینال باز می شود استفاده شود .
</div>


[[https://www.realvnc.com/en/connect/download/viewer/]{.underline}](https://www.realvnc.com/en/connect/download/viewer/)

connection {#connection .ListParagraph}
----------

<div dir="rtl">

ping raspberrypi.local → shows ip... ip of this board is 192.168.43.22
</div>


ssh [[pi\@raspberrypi.local]{.underline}](mailto:pi@raspberrypi.local)
pass: raspberry

<div dir="rtl">

تفاوت compile build **link** ...
</div>

================================

<div dir="rtl">

وقتی مثلا به زبان C کدنویسی می کنید، کامپایلر دستورات این زبان را به
زبان ماشین تبدیل می کند(instructionهایی که توسط cpu قابل فهم است.)
</div>


<div dir="rtl">

اما معمولا دستورات در یک فایل .c نیستند و با فایل های .h آن ها را
فراخوانی می کنیم. dll؟؟؟
</div>


<div dir="rtl">

وقتی مثلا با visual studio ؟ کار می کنید در فایلی مربوط به wrokspace ؟
آدرس این فایل ها را می نویسد.
</div>


![](.//media/image34.png){width="4.74375in"
height="2.8194444444444446in"}

