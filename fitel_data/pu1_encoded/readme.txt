This directory contains the PU1 corpus, as described in the paper:

I. Androutsopoulos, J. Koutsias, K.V. Chandrinos, and C.D. Spyropoulos, 
"An Experimental Comparison of Naive Bayesian and Keyword-Based 
Anti-Spam Filtering with Personal E-mail Messages". In Belkin, N.J., 
Ingwersen, P. and Leong, M.-K. (Eds.), Proceedings of the 23rd 
Annual International ACM SIGIR Conference on Research and Development 
in Information Retrieval (SIGIR 2000), Athens, Greece, pp. 160-167, 
2000.

There are 4 subdirectories, corresponding to the four "encrypted" 
versions of the corpus mentioned in the paper:

bare: Lemmatiser disabled, stop-list disabled.
lemm: Lemmatiser enabled, stop-list disabled.
lemm_stop: Lemmatiser enabled, stop-list enabled.
stop: Lemmatiser disabled, stop-list enabled.

Each one of these 4 directories contains 10 subdirectories (part1, ...,
part10). These correspond to the 10 partitions of the corpus that 
were used in the 10-fold experiments. In each repetition, one part
was reserved for testing and the other 9 were used for training. 

Each one of the 10 subdirectories contains both spam and legitimate 
messages, one message in each file. Files whose names have the form
*spmsg*.txt are spam messages. Files whose names have the form 
*legit*.txt are legitimate messages. 

You are free to use this corpus for non-commercial purposes, provided 
that you acknowledge the use and origin of the corpus in any published
work of yours that makes use of the corpus, and that you notify the 
person below about this work. To use this corpus for commercial 
applications, you must obtain a written permission from the person below.

Ion Androutsopoulos 
http://www.aueb.gr/users/ion/

PU1 corpus last updated: July 17, 2000.
This file (readme.txt) last updated: July 30, 2003.

