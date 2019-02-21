# Doubly-Stochastic-DGP
Deep Gaussian Processes with Doubly Stochastic Variational Inference 

Requirements: gpflow1.1.1 and tensorflow1.8. NB not compatabile with more recent versions (e.g. gpflow1.2)

This code accompanies the paper

@inproceedings{salimbeni2017doubly,
  title={Doubly stochastic variational inference for deep gaussian processes},
  author={Salimbeni, Hugh and Deisenroth, Marc},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}

See the arxiv version at https://arxiv.org/abs/1705.08933

This code now offers additional functionality than in the above paper. In particular, natural gradients are now supported. If you use these, please consider citing the following paper:

@inproceedings{salimbeni2018natural,
  title={Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models},
  author={Salimbeni, Hugh and Eleftheriadis, Stefanos and Hensman, James},
  booktitle={Artificial Intelligence and Statistics},
  year={2018}
}
