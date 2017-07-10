import pandas as pd
import numpy as np
import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
#essentiall
def pre_processing(raw_text):
	stops= set(stopwords.words("english"))
	html_removed=BeautifulSoup(raw_text,"html.parser").get_text()
	letters_only=re.sub("[^a-zA-Z]", " ", html_removed).lower().split()
	remove_stops=[w for w in letters_only if not w in stops]
	remove_space=" ".join( remove_stops)
	return remove_space
def train_blackforest_sentiment(path):
	#trains on the file labeledTrainData.tsv
	#header=0 saying that the first line of the file has the column names
	#delimited="\t" means that all the fields are separated by tabs
	#quoting=3 means that you ignore double quotes
	traindata=pd.read_csv(path, header=0, delimiter="\t", quoting=3)
	print ("Pre-processing our training set...\n")
	pre_processed_reviews=[]
	i=1000
	for review in traindata["review"]:
		pre_processed_reviews.append(pre_processing(review))
		if(i%1000==0):
			print ("On review " + str(i))
		i+=1
	print ("Creating BOW..\n")
	#Initializes CountVectorizer object which is a bag of words tool
	vectorizer= CountVectorizer(analyzer ="word",tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)

	#fits the model and learns vocab
	#fit_transform does 2 things
	#1.It finds the mean and standard devation and centers the data and saves that
	#2 Transforms applies the tranformations
	#also transforms training data into feature vectors aka vectors with the word and how many there are
	#arrays are easier to work with so we convert from list of strings into an array
	train_data_features=vectorizer.fit_transform(pre_processed_reviews).toarray()
	#print (train_data_features.shape)

	#vocab=vectorizer.get_feature_names()
	# print (vocab)
	#dist=np.sum(train_data_features, axis=0)

	# for tag, count in zip(vocab, dist):
	# 	print (count, tag)

	print ("Training the random forest...\n")
	#Makes a Random Forest Classifier with 100 trees
	forest=RandomForestClassifier(n_estimators=100)
	#fits the forest to the training set usinga  bag of words as features and the sentiment labels as the response variable
	forest=forest.fit(train_data_features, traindata["sentiment"])	
	return forest,vectorizer
def forest_predict(path,forest,vectorizer):
	test=pd.read_csv(path, header=0, delimiter="\t", quoting=3)

	num_reviews=len(test["review"])
	pre_processed_reviews=[]


	print ("Pre-processing our test set...\n")
	i=1000
	for review in test["review"]:
		pre_processed_reviews.append(pre_processing(review))
		if(i%1000==0):
			print ("On review " + str(i))
		i+=1

	#converts to the special numpy array just like in 
	test_data_features=vectorizer.transform(pre_processed_reviews).toarray()

	#the actual predicting part
	result=forest.predict(test_data_features)
	output=pd.DataFrame( data={"id":test["id"], "sentiment":result} )
	return output

	#Use pandas to write the comma-separated output file

if __name__ == '__main__':
	forest_sentiment,vectorizer=train_blackforest_sentiment("/Users/bhuang/Desktop/bagsof(words and popcorns)/data/labeledTrainData.tsv")
	output=forest_predict("/Users/bhuang/Desktop/bagsof(words and popcorns)/data/testData.tsv",forest_sentiment,vectorizer)
	
	output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3)
