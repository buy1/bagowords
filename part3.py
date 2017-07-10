from gensim.models import Word2Vec
from gensim import models
import numpy as np



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

def getAverageFeatures(reviews, model, num_features):
	counter=0
	reviewFeatureVecs=np.zeros((len))


# model=Word2Vec.load('300features_40minwords_10context.bin')
model=models.KeyedVectors.load_word2vec_format("glove_word2vec.bin")

print (model['car'])