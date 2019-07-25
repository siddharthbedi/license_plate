
```

## Requirements

In order to easily run the code, you must have installed the Keras framework with TensorFlow backend. The Darknet framework is self-contained in the "darknet" folder and must be compiled before running the tests. To build Darknet just type "make" in "darknet" folder:

```shellscript
$ cd darknet && make
```


## Download Models

After building the Darknet framework, you must execute the "get-networks.sh" script. This will download all the trained models:

```shellscript
$ bash get-networks.sh
```
# Test run using

 python predict.py -c config.json -w {trained weights path} -i ./images/test{'i'}.mp4 
