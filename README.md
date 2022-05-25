# NeuTic

NeuTic is a deep neural network (DNN) model for traffic classification. It contains a multi-branch convolutional structure and the self-attention mechanism.

Note: the self-attention mechanism in NeuTic is based on [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch). Many thanks to the authors.

## Usage

Each item `x`  in the input data of NeuTic should contiains three lists as follows:

>x = (\[len<sub>1</sub>, len<sub>2</sub>, ..., len<sub>h</sub>\], \[win<sub>1</sub>, win<sub>2</sub>, ..., win<sub>h</sub>\], \[flag<sub>1</sub>, flag<sub>2</sub>, ..., flag<sub>h</sub>\])

`h` denotes the length of the sequence.