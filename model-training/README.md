# Haggis Recognition Model Training

This repo contains a script to train a state-of-the-art image recognition model to detect a haggis using Keras, Tensorflow and FloydHub.

To train the model on a GPU on FloydHub, simply clone this repository locally, install the `floyd-cli`, initialize a project on FloydHub and type the following command into your terminal:

```$ floyd run --env tensorflow-1.12 --gpu --data euanwielewski/datasets/haggis-dataset/1:data "python training.py"```

Alternatively, you can just click the button below to automatically run the job on FloydHub:

[![Run on FloydHub](https://static.floydhub.com/button/button-small.svg)](https://floydhub.com/run)