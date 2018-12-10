Google_Image_Classification
==============================

Playing around with neural networks and image classification. 


## Transfer learning on Google Images
Downloads ~200 Google photos for two specified things (e.g. "Snake" and "Moray Eel", or "Aerial Farmland" and "Green Quilt Pattern")
and uses transfer learning (replacing the last dense layer of the Resnet50 model) with early stopping to learn to 
differentiate these two things. Often achieves >95% accuracy in the validation images. 
https://github.com/zwrankin/google_image_classification/blob/master/notebooks/2018_12_09_transfer_learning_classify_google_images_snake_vs_moray_eel.ipynb
**Example**
![Alt text](reports/example_classification.jpg?raw=true "Grizzly Bear vs Teddy Bear")

## Resnet50 classification 
For comic relief, use trained ResNet50 model to classify my own photographs
https://github.com/zwrankin/google_image_classification/blob/master/notebooks/2018_12_09_resnet50_image_prediction.ipynb


### TODO 
- Utilities to compress images (not a big deal for the scale we're working at)
- Use `click` to make the tool a command-line utility (right now just interactive in notebooks)
- Utilities to generate standardized reports from the trained models (for now just in notebooks)
- Multiclass classification? Maybe beyond the scope of this toy repo

Many tools adapted from: 
- https://www.kaggle.com/dansbecker/programming-in-tensorflow-and-keras
- https://www.kaggle.com/dansbecker/transfer-learning

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          
    ├── data
    │   ├── processed      <- (empty)
    │   └── raw            <- Where google images are downloaded
    │
    ├── docs               <- (empty)
    │
    ├── models             <- Trained models
    │
    ├── notebooks          <- Jupyter notebooks showing modeling process 
    │
    ├── reports            <- (empty)
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment,
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── download_data.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── resnet50_model.py
    │   │   └── transfer_learning_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
