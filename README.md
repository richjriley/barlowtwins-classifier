# Barlow Twins Classifier

This repository is a demonstration of how to utilise Facebook's pre-trained Barlow Twins model for a simple classification task.

The original Barlow Twins paper can be found here: https://arxiv.org/abs/2103.03230

This code was written to accompany the following blog post: [INSERT URL] 

# Install Dependencies
From the root of this repo: \
```conda env create``` \
```conda activate barlowtwins-classifier```

# Download Data
The data used in the accompanying blog post can be downloaded from here:
https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset

Then unzip to the 'data' directory of this repo: \
```cd data``` \
```unzip archive.zip```

# Extract Embeddings
Each image in the dataset is fed through the Barlow Twins model to obtain the embeddings. The embeddings are stored in a csv file.

Warning this takes approximately 1 hour on a CPU.

```python3 extract_embeddings.csv```

# Train Model
To train the model: \
```python3 train.py```