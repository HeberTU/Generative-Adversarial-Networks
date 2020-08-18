# Generative-Adversarial-Networks

This python package provides a general framework to generate small images (e.g. 28 x 28 pixels) using a generative adversarial network (GAN).

We have been using the general theoretical background provided by Ian Goodfellow on his [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) research paper.

## Intuition

The idea behind GANs is that two networks, a generator  𝐺, and a discriminator  𝐷, will compete against each other. The generator makes "fake" data to pass to the discriminator. The discriminator also sees real training data and predicts if the data it's received is real or fake.

The following diagram shows the general idea:

![image](https://user-images.githubusercontent.com/28582065/90556171-bbc8d080-e198-11ea-88ba-9beb3ef8cef6.png)

This way we will be end-up with a Generator capable of generate new training data from scratch.
