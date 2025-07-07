a face recognition model training script USING TENSORFLOW
now trying to move to pytorch for this script, also added 
exporting the embeddings and saving the model in the pytorch 
version. Training has not been done YET to verify and debug the
model.

The model is trained on the Labeled Faces in the Wild (LFW) image dataset.
It is a public dataset that I used to train the model, and you can plug in 
and use your own, just make sure you change the variable for the images path 
```Python image_base_dir = 'archive/images/images'```

Other stuff is pretty straight forward just clone the repo and it will work!
