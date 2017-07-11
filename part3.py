from gensim.models import Word2Vec
from gensim import models
from bs4 import BeautifulSoup
import numpy as np
import os
import pandas as pd
from nltk.corpus import stopwords
import nltk.data
import logging
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import time
def create_bag_of_centroids( wordlist, word_centroid_map ):

    num_centroids = max( word_centroid_map.values() ) + 1    #
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )

    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1

    return bag_of_centroids
#Takes in a raw_String and removes html elements and non alphabetical characters
#returns a list of words
def review_to_wordlist(review, remove_stopwords=False):
	#Function to convert document to a sequence of words and optionally remove stopwords

	#removes html elements and alphabet words
	review_text=re.sub("[^a-zA-Z]"," ",BeautifulSoup(review,"html.parser").get_text())
	#puts the words in lowercase and tokenizes it by whitespace
	words= review_text.lower().split()
	#optionally if remove_stopwords is true
	if remove_stopwords:
		#changes the array into a set
		stops=set(stopwords.words("english"))
		#removes the stopwords
		words=[w for w in words if not w in stops]
	return (words)

#All the data has variable length so we try averaging the vectors for each review
#word is the review
def makeFeatureVec(words, model, num_features,index2word_set):
	#pre-initialize for speed
	featureVec=np.zeros((num_features),dtype="float32")

	# checks if the words in the review are in the model's vocab
	# and adds its feature vector to the total
	nwords=0
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
	index2word_set=set(model.index2word)
	for review in reviews:
		if counter%1000:
			print ("On review : " + str(counter))
		reviewFeatureVecs[counter]= makeFeatureVec(review,model,num_features,index2word_set)
		counter+=1
	return reviewFeatureVecs

# model=Word2Vec.load('300features_40minwords_10context.bin')
labeledpath=os.getcwd()+"/data/labeledTrainData.tsv"
testpath=os.getcwd()+"/data/testData.tsv"
# unlabeled_path=os.getcwd()+"data/unlabeledTrainData.tsv"

train=pd.read_csv(labeledpath, header=0,delimiter="\t", quoting=3)
test=pd.read_csv(testpath,header=0,delimiter="\t",quoting=3)
# unlabeled_traindata=pd.read_csv(unlabeled_path,header=0,delimiter="\t", quoting=3)
#open('u.item', encoding = "ISO-8859-1")
model=models.KeyedVectors.load_word2vec_format("/mnt/home/huangbaiwen/model/glove-word2vec.bin",binary=True)
num_features=300

print ("Creating average feature vecs for train reviews")
clean_train_reviews=[]
for review in train["review"]:
	clean_train_reviews.append(review_to_wordlist(review,remove_stopwords=True))
trainDataVecs= getAvgFeatureVecs(clean_train_reviews,model, num_features)

print ("Creating average feature vecs for test reviews")
clean_test_reviews=[]
for review in test["review"]:
	clean_test_reviews.append(review_to_wordlist(review,remove_stopwords=True))
testDataVecs=getAvgFeatureVecs(clean_test_reviews,model,num_features)

# forest = RandomForestClassifier(n_estimators=100)

# print ("fitting a random forest to labeled training data...")
# forest = forest.fit(trainDataVecs,train["sentiment"])

# result=forest.predict(testDataVecs)

# output=pd.DataFrame(data={"id":test["id"],"sentiment":result})
# output.to_csv("Word2Vec_AverageVectors.csv",index=False, quoting=3)

start = time.time() # Start time

#syn0: a numpy array that stores a word with all its features in each row
word_vectors = model.syn0
#sets the number of clusters to be 1/5th the size of words
# aka 5 words per cluster instead of just shape[0] which would be 5 words per cluster
num_clusters = word_vectors.shape[0] / 5

#initialize the the trainer object
kmeans_clustering = KMeans( n_clusters = num_clusters )
# fits the model to the data
idx = kmeans_clustering.fit_predict( word_vectors )

# Get the end time and print how long the process took
end = time.time()

elapsed = end - start
print "Time taken for K Means clustering: ", elapsed, "seconds."
num_clusters=word_vectors.shape[0]/5

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number                                                              
word_centroid_map = dict(zip( model.index2word, idx ))

# For the first 10 clusters
for cluster in xrange(0,10):
    #
    # Print the cluster number  
    print "\nCluster %d" % cluster
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    for i in xrange(0,len(word_centroid_map.values())):
        if( word_centroid_map.values()[i] == cluster ):
            words.append(word_centroid_map.keys()[i])
    print words

