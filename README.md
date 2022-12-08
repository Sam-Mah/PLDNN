# PLDNN
This repository is an implementation of the paper "Dynamic Android Malware Category Classification
using Semi-Supervised Deep Learning" (https://ieeexplore.ieee.org/abstract/document/9251198).

In this paper, we proposed an effective and efficient Android malware category classification system based on semi-supervised deep neural networks. In spite of the small
number of labeled training samples, the proposed detection system is effective and superior to supervised deep neural networks. This eliminates the need for a high number of labeled
instances, which is very expensive to acquire in the domain of malware analysis.

 We use dynamic analysis to craft dynamic behavior profiles as feature vectors. Furthermore, we develop a new dataset, namely CICMalDroid2020, which includes 17,341 most recent samples of five different Android apps categories: Adware, Banking, SMS, Riskware, and Benign. Our offered dataset comprises the most complete captured static and dynamic
features among publicly available datasets (https://www.unb.ca/cic/datasets/maldroid-2020.html).

Our model, called Pseudo-Label Deep Neural Network (PLDNN), significantly outperforms Label Propagation (LP), which is a popular semi-supervised machine learning algorithm, and other common machine learning algorithms such as Random Forest (RF), Decision Tree (DT), Support Vector Machine (SVM), and k-Nearest Neighbor (k-NN).

This project is implemented using Python and Tensorflow library.
