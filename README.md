# simple_genre_discriminator
collect anime images from miscellaneous images in the web

## just predict !

```
python predict.py weights.2251-2.63-0.84.hdf5 your_image.png
```

## train

your dataset directory should be like...

* path/to/dataset/directory
  * anime
  * others

(directory names can be anything you like)

```
python simple_genre_discriminator/train.py path/to/dataset/directory --optimizer adam --freeze_vgg
```
