# SV-softmax
Tensorflow implementation of the Support Vector Guided Softmax Loss for Face Recognition paper (https://arxiv.org/pdf/1812.11317.pdf).

# Details
You can see the loss implementation details and related math in [this](https://github.com/comratvlad/sv_softmax/blob/master/notebooks/check_maths_release.ipynb) notebook. It's better to watch it on your local computer than on github because of the problems with the display of the notebooks. You can also fing losses [here](https://github.com/comratvlad/sv_softmax/blob/master/src/custom_losses.py).

# Results
We use cifar-10 and cifar-100 to validate implementation and very simple cnn-model.

Loss                      | cifar-10   | cifar-100  
--------------------------|------------|-----------
softmax                   | 0.7729     |  0.4917     
sv-softmax, t=1.05        | 0.8059     |  0.4773   
sv-softmax, t=1.1         | 0.8171     |  0.5122     
sv-softmax, t=1.2         | **0.8219** |  **0.5225**   
sv-softmax, t=1.4         | 0.8184     |  0.5158    

# Future work
SV-AM-softmax doesn't show improvement on the cifar-10 and cifar-100 datasets. Maybe it's better works in Face Recognition problem, so it's a good idea to test it on such problems.
