# Tensorflow models with simple ways
Some models with easy understanding code,it will help you understand what the model does.These models can only work with not very good results.
##  Models list 
1.Adversarial Networks (GAN)<br>
* AAE      &nbsp;&nbsp;[Adversarial Autoencoders](https://arxiv.org/abs/1511.05644) <br> 
* ACGAN    &nbsp;&nbsp;[Conditional Image Synthesis With Auxiliary Classifier GANs](https://arxiv.org/abs/1610.09585)<br>
* Auto encoder&nbsp;&nbsp;[Recent Advances in Autoencoder-Based Representation Learning](https://arxiv.org/abs/1812.05069)<br>
* BGAN     &nbsp;&nbsp;[Boundary-Seeking Generative Adversarial Networks](https://arxiv.org/abs/1702.08431)<br>
* BiGAN    &nbsp;&nbsp;[Bidirectional Generative Adversarial Network](https://arxiv.org/abs/1605.09782)<br>
* CCGAN    &nbsp;&nbsp;[Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks](https://arxiv.org/abs/1611.06430)<br>
* CGAN     &nbsp;&nbsp;[Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)<br>
* CoGAN    &nbsp;&nbsp;[Coupled generative adversarial networks](https://arxiv.org/abs/1606.07536)<br>
* CycleGAN &nbsp;&nbsp;[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)<br>
* DCGAN    &nbsp;&nbsp;[Deep Convolutional Generative Adversarial Network](https://arxiv.org/abs/1511.06434)<br>
* GAN      &nbsp;&nbsp;[Generative Adversarial Network with a MLP generator and discriminator](https://arxiv.org/abs/1406.2661)<br>
* VAE      &nbsp;&nbsp;[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)<br>

2.Special structure Convolutional network (Not recurring paper)<br>
* DenseNet            &nbsp;&nbsp;[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)<br>
* ResNet              &nbsp;&nbsp;[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)<br>
* HighwayNet          &nbsp;&nbsp;[Highway Networks](https://arxiv.org/abs/1505.00387)<br>
* MobileNet_v1&v2     &nbsp;&nbsp;[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)<br>
 &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)<br>

3.Other Models<br>
* None<br>

## Results of some models
The results are based on MNIST or Fashion-MNIST

### AAE
<img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/aae_Adversarial%20Autoencoders/24000_prediction.png"/><img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/aae_Adversarial%20Autoencoders/24000.png"/>

### ACGAN
<img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/acgan_/6000.png"/><img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/acgan_/9000.png"/>

### Auto_encoder
Input image &rarr; Hidden layer &rarr;  Output image <br>
<img width="280" height="280" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/auto_encoder/10000_real.png"/><img width="280" height="280" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/auto_encoder/10000_prediction.png"/><img width="280" height="280" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/auto_encoder/10000_fake.png"/>

### BGAN
<img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/bgan/12000.png"/>

### BiGAN
<img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/bigan/5000.png"/><img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/bigan/5000_prediction.png"/>

### CCGAN
Real image &rarr; Random cropping image &rarr;  Repaired image <br>
<img width="280" height="280" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/ccgan/19000_real.png"/><img width="280" height="280" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/ccgan/19000_mis.png"/><img width="280" height="280" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/ccgan/19000_rebuild.png"/>

### CGAN
<img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/cgan/20000.png"/>

### CoGAN
A: MNIST &nbsp;&nbsp;&nbsp;&nbsp;   B: Rotate 90 degrees MNIST <br>
The model try to convert between A and B.<br>
A &rarr; B <br>
<img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/cogan/12000_1_real.png"/><img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/cogan/12000_1_fake.png"/><br>
B &rarr; A <br>
<img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/cogan/12000_2_real.png"/><img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/cogan/12000_2_fake.png"/>

### CycleGAN
A: MNIST &nbsp;&nbsp;&nbsp;&nbsp;   B: Rotate 90 degrees MNIST <br>
The model try to convert between A and B.<br>
A &rarr; B <br>
<img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/cyclegan/16000_real_a.png"/><img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/cyclegan/16000_fake_a2b.png"/><br>
B &rarr; A <br>
<img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/cyclegan/16000_real_b.png"/><img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/cyclegan/16000_fake_b2a.png"/>

### DCGAN
<img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/dcgan/6000.png"/>

### GAN
<img width="300" height="300" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/gan/99000.png"/>

### VAE
Input image &rarr; Hidden layer &rarr;  Output image <br>
<img width="280" height="280" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/vae/11000_real.png"/><img width="280" height="280" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/vae/11000_prediction.png"/><img width="280" height="280" src="https://github.com/Taoerwang/tensorflow_model_with_simple_ways/raw/master/vae/11000_fake.png"/>
