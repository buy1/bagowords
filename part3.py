# from gensim.models import Word2Vec
# from gensim import models
import gensim.models as models
import numpy as np
import os
import pandas as pd

#All the data has variable length so we try averaging the vectors for each review
#word is the review
def makeFeatureVec(words, model, num_features):
	#pre-initialize for speed
	featureVec=np.zeros((num_features),dtype="float32")

	#converts the model to a list of words in the model's vocab
	index2word_set=set(model.index2word)

	# checks if the words in the review are in the model's vocab
	# and adds its feature vector to the total
	n_words=0
	for word in words:
		if word in index2word_set:
			featureVec=np.add(featureVec,model[word])
			#model[word] gives you the feature vector for that word
			nwords+=1

	#divide by the number of words to get the average
	featureVec=np.divide(featureVec,nwords)
	return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
	reviewFeatureVecs=np.zeros((len(reviews),num_features),dtype="float32")

	counter=0
	for review in reviews:
		if counter%1000:
			print ("On review : " + str(counter))
		reviewFeatureVecs[counter]= makeFeaturevec(revew,model,num_features)
		counter=counter+1
	return reviewFeatureVecs

# model=Word2Vec.load('300features_40minwords_10context.bin')
labeledpath=os.getcwd()+"/data/labeledTrainData.tsv"
testpath=os.getcwd()+"/data/testData.tsv"
# unlabeled_path=os.getcwd()+"data/unlabeledTrainData.tsv"

traindata=pd.read_csv(labeledpath, header=0,delimiter="\t", quoting=3)
testdata=pd.read_csv(testpath,header=0,delimiter="\t",quoting=3)
# unlabeled_traindata=pd.read_csv(unlabeled_path,header=0,delimiter="\t", quoting=3)
#open('u.item', encoding = "ISO-8859-1")
model=models.KeyedVectors.load_word2vec_format("/mnt/home/huangbaiwen/model/glove-word2vec.bin",binary=True)
num_features=300

print ("Creating average feature vecs for train reviews")
clean_train_reviews=[]
for review in traindata["review"]:
	clean_train_reviews.append(review_to_wordlist(review,remove_stopwords=True))
trainDataVecs= getAvgFeatureVecs(clean_train_reviews,model, num_features)

print ("Creating average feature vecs for test reviews")
clean_test_reviews=[]
for review in testdata["review"]:
	clean_test_reviews.append(review_to_worldlist(review,remove_stopwords=True))
testDataVecs=getAvgFeatureVecs(clean_test_reviews,model,num_features)


