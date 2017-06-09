This code uses fastText supervised learning to predict output labels from input text.

Approach

This is the baseline code. I have not changed anything.

Preprocessing

I've done no preprocessing.

Parameters

I've changed components as follows:
1. size of word vectors - 50
2. minimal number of word occures - 2
3. maximaxl length of character ngram - 4
4. size of the context window - 3
5. maximal length of word ngram - 2

Some part of these were done to decrease the running time of programm.

Results

Precision and recall were 0.937.