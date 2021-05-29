### Tasks May 29
1. Plot Word2vec embeddings
2. Tool
3. TSNE
4. PCA
5. SOM
6. Print which tweets are correct
### Tasks May 27
1. Hemanth's Data
2. Accuracy with different splits
### Tasks May 16
4. Word2vec for all the words to capture semantic meaning. Cosine Similarity->Normalization (check first). Decide Threshold?
5. BERT base model and BERT base +
6. Overfitting
7. 80-20 split
### Tasks May 14
1. Identity Matrix vs Feature Matrix
2. Efficient Graph Creation from tweets instead of TF-IDF: Capture Importance of words in a tweet
3. VGCN Paper
4. Graph Encoders for embedding generation
5. Write to WNUT paper authors
6. BERT Tokenizer
### Tasks May 10
1. Try new models like ALBERT
2. Explainable BERT, DistilBERT
3. BERT Tweet
4. GCN experiment
5. Activation Functions : SELU
6. Loss Function: Hinge Loss, Normalised Log likelihood
![image](https://user-images.githubusercontent.com/48908329/118227602-7080eb80-b4a6-11eb-8d02-673479b3d6df.png)
![image](https://user-images.githubusercontent.com/48908329/118227623-7a0a5380-b4a6-11eb-8f2b-ff6db5ac7b92.png)

### Notes and tasks - May 7
1. Perform an experiment on time for build graph: For 500 tweets, time to build graph = 26.051642417907715 seconds; For 600 tweets, time to build graph = 31.93747115135193 seconds, For 2000 tweets, time to build graph = 64.18152785301208 seconds
2. Add Papers to the file
3. Make this repository private if possible
4. Remove shuffling in the build graph file
5. Discuss about XAI techniques for BERT
6. Minor Changes to code structure
7. Exception Handling
### Notes - April 24
#### Data
We have 7000 tweets which we split as train and test in most of the models. We have additional 2000 tweets which are untouched at the moment
#### Experiments
1. Text GCN: 5000:2000, 79.4 
	- 500: 6500, 71% 
2. Feature embeddings are obtained from tf-idf of the synonym set of words
3. BERT: Base, train = 93% (3 epochs), test = 87%
4. Embeddings from Text - GCN and BERT both, Multilayer Perceptron Test: 90%
	- This means that we took the embeddings for both the models of size 200 for GCN and 768 for BERT and concatenated them before feeding into MLP.
5. Not done yet: Using BERT embeddings as GCN features because it is time intensive to generate bert embeddings for context free individual words.
6. Text GCN is transductive
	- (AXW) 7000*7000 7000 * feature_size * Feature_size * hidden_layer
	- X'=(AXW)
	- Fast GCN, which is inductive in nature

## Requirements
1. pip3 install -r requirements.txt
2. download bert pre_trained models from [https://drive.google.com/drive/folders/1rZmeT5SCCLe7UH6SScXj7aaN_f8TD8gs?usp=sharing] and store into the folder '../BERT/'
	- finetuned_BERT_epoch_1.model
	- finetuned_BERT_epoch_2.model
	- finetuned_BERT_epoch_3.model

## Steps to run the code:
1. Install all the requirements in the file requirements.txt by using the above code.
2. python run.py
3. run.py has commands to run 6 python files:
	- preprocess.py: file to change for any new data and preprocess the tweets.
	- BERT_evaluate.py: file to generate BERT embeddings from a pre-trained and finetuned model.
	- prepare_data.py: prepare data for graph convolutional neural networks.
	- build_graph.py: build a text graph.
	- gcn_train.py: train GCN and generate embeddings.
	- mlp_twignet.py: run mlp on the concatenated and generated embeddings.

## Code to be added:
1. BERT_train.py: File to train BERT
2. GCN_evaluate.py: Have a pre-generated adjacency and feature matrix for the train set in GCN and write code to use the additional actual data to enlarge the graph as needed without recreating the original graph for the nodes in the training set. 
	
