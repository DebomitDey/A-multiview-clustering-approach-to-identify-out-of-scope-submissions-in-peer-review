import os
import re 
import sys
import json
import nltk
import rake
import math
import numpy as np
from nltk import pos_tag
from scipy import spatial
from pyemd import emd
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from numpy import linalg as LA
from sklearn.decomposition import PCA
from operator import sub,mul
from scipy.spatial import distance
# from nsga2.evolution import Evolution
alpha=0.001
def cosin(temp1,temp2,temp):
	dist=0.0
	#dist = math.fabs(np.dot(temp1,temp2)/(alpha+math.sqrt(LA.norm(temp1)*LA.norm(temp2))))
	#dist=1.00-dist
	#temp1=list(map(sub,temp1,temp2))
	#dist=math.sqrt(sum(list(map(mul,temp1,temp1))))
	dist=distance.mahalanobis(temp1,temp2,temp)
	return dist

base = "/home/debomit/Desktop/hh/"
base1 = "Science_Parse_json/ARTINT_ACC/artint_acc_500_json/"
base2 = "DATA/Train/"
base3 = "DATA/Test/InScope/"
base4 = "DATA/Test/OutScope/"



#print len(model['king'])

reload(sys) 
sys.setdefaultencoding('UTF8')

stopwords_ = []

with open(base+"SmartStoplist.txt","r")as in_file:
	for lines in in_file:
		stopwords_.append(lines.strip())

def get_continuous_chunks(text):
	chunked = ne_chunk(pos_tag(word_tokenize(text)))
	prev = None
	continuous_chunk = []
	current_chunk = []
	for i in chunked:
		if type(i) == Tree:
			current_chunk.append(" ".join([token for token, pos in i.leaves()]))
		elif current_chunk:
			named_entity = " ".join(current_chunk)
			if named_entity not in continuous_chunk:
				continuous_chunk.append(named_entity)
				current_chunk = []
			else:
				continue
	return continuous_chunk		
	

# =======================  Corpus Creation  =====================================

delimiters = ['\n', ' ', ',', '.', '?', '!', ':']


########  Training Corpus ##############
journal_corpus = []
text_corpus = []
venue_corpus = []
author_corpus = []
title_corpus = []

venue_vocab = []
author_vocab = []
title_vocab = []

Capitalised = []
POS_tag = []
NER = []
RAKE = [] 

count = 0

print("Traing Corpus") 

for file in os.listdir(base+base2):
	count += 1
	print(str(count)+" "+file)
	try:	
		with open(base+base2+file,"r") as input_file:			
			text = ""
			venue = ""
			author = ""
			title = ""
			data = json.load(input_file)
			lexical = data['metadata']['sections']
			bib_venue = data['metadata']['references']
			if lexical is not None:
				for i in range(len(lexical)):
					text += lexical[i]['text']
			text=re.sub('[^A-Za-z]+',' ',text).strip()
			if bib_venue is not None:
				for i in range(len(bib_venue)):
					venue += str(bib_venue[i]['venue'])+" "
					# author = ' '.join(bib_venue[i]['author'])
					title += bib_venue[i]['title']+" "	
			venue=re.sub('[^A-Za-z]+',' ',venue).strip()
			title=re.sub('[^A-Za-z]+',' ',title).strip()
			# text_corpus.append(text)
			venue_corpus.append(venue)	
			# author_corpus.append(author)
			title_corpus.append(title)
			tokens = wordpunct_tokenize(text)
			named_entities = get_continuous_chunks(text)
			rake_obj = rake.Rake(base+"SmartStoplist.txt")
			rake_keywords = rake_obj.run(text)
			rake_keywords = sorted(rake_keywords,key= lambda s : s[1],reverse = True)
			rake_keywords = [i[0] for i in rake_keywords]
			rake_keywords_lex = rake_keywords
			rake_keywords = rake_keywords[0:30]	
			rake_keywords_lex = " ".join(rake_keywords_lex)
			rake_keywords_lex=re.sub('[^A-Za-z]+',' ',rake_keywords_lex).strip()
			#print(rake_keywords_lex)
			#print(type(rake_keywords_lex))
			text_corpus.append(rake_keywords_lex)
			NER.append(named_entities)
			RAKE.append(rake_keywords)
			#NER.append(re.sub('[^A-Za-z]+',' ',named_entities).strip())
			#RAKE.append(re.sub('[^A-Za-z]+',' ',rake_keywords).strip())
			#print(text)
			lex_tfidf = named_entities + rake_keywords
	except:
		count=count-1
########  Test Corpus IN ##############

text_corpus_test = []
venue_corpus_test = []
title_corpus_test = []

NER_test = []
RAKE_test = [] 

count_in = 0

print("Test Corpus IN")

for file in os.listdir(base+base3):
	count_in += 1
	print(str(count_in)+" "+file)
	try:
		with open(base+base3+file,"r") as input_file:
			text = ""
			venue = ""
			author = ""
			title = ""
			data = json.load(input_file)
			lexical = data['metadata']['sections']
			bib_venue = data['metadata']['references']
			if lexical is not None:
				for i in range(len(lexical)):
					text += lexical[i]['text']
			text=re.sub('[^A-Za-z]+',' ',text).strip()
			if bib_venue is not None:
				for i in range(len(bib_venue)):
					venue += str(bib_venue[i]['venue'])+" "
					title += bib_venue[i]['title']+" "
			venue=re.sub('[^A-Za-z]+',' ',venue).strip()
			title=re.sub('[^A-Za-z]+',' ',title).strip()
			# text_corpus_test.append(text)
			venue_corpus_test.append(venue)	
			title_corpus_test.append(title)
			tokens = wordpunct_tokenize(text)
			named_entities = get_continuous_chunks(text)
			rake_obj = rake.Rake(base+"SmartStoplist.txt")
			rake_keywords = rake_obj.run(text)
			rake_keywords = sorted(rake_keywords,key= lambda s : s[1],reverse = True)
			rake_keywords = [i[0] for i in rake_keywords]
			rake_keywords_lex = rake_keywords
			rake_keywords = rake_keywords[0:30]
			rake_keywords_lex = " ".join(rake_keywords_lex)
			rake_keywords_lex=re.sub('[^A-Za-z]+',' ',rake_keywords_lex).strip()
			text_corpus_test.append(rake_keywords_lex)
			NER_test.append(named_entities)
			RAKE_test.append(rake_keywords)
			#NER_test.append(re.sub('[^A-Za-z]+',' ',named_entities).strip())
			#RAKE_test.append(re.sub('[^A-Za-z]+',' ',rake_keywords).strip())
	except:
		count_in=count_in-1			
	# if(count_in == 5):
	# 	break

########  Test Corpus OUT ##############

text_corpus_test_out = []
venue_corpus_test_out = []
title_corpus_test_out = []

NER_test_out = []
RAKE_test_out = [] 

count_out = 0

print("Test Corpus OUT")

for file in os.listdir(base+base4):
	count_out += 1
	print(str(count_out)+" "+file)
	try:
		with open(base+base4+file,"r") as input_file:
			text = ""
			venue = ""
			author = ""
			title = ""
			data = json.load(input_file)
			lexical =data['metadata']['sections']
			bib_venue = data['metadata']['references']
			if lexical is not None:
				for i in range(len(lexical)):
					text += lexical[i]['text']
			text=re.sub('[^A-Za-z]+',' ',text).strip()
			if bib_venue is not None:
				for i in range(len(bib_venue)):
					venue += str(bib_venue[i]['venue'])+" "
					title += bib_venue[i]['title']+" "
			venue=re.sub('[^A-Za-z]+',' ',venue).strip()
			title=re.sub('[^A-Za-z]+',' ',title).strip()
			# text_corpus_test.append(text)
			venue_corpus_test_out.append(venue)	
			title_corpus_test_out.append(title)
			tokens = wordpunct_tokenize(text)
			named_entities = get_continuous_chunks(text)
			rake_obj = rake.Rake(base+"SmartStoplist.txt")
			rake_keywords = rake_obj.run(text)
			rake_keywords = sorted(rake_keywords,key= lambda s : s[1],reverse = True)
			rake_keywords = [i[0] for i in rake_keywords]
			rake_keywords_lex = rake_keywords
			rake_keywords = rake_keywords[0:30]
			rake_keywords_lex = " ".join(rake_keywords_lex)
			rake_keywords_lex=re.sub('[^A-Za-z]+',' ',rake_keywords_lex).strip()
			text_corpus_test_out.append(rake_keywords_lex)
			NER_test_out.append(named_entities)
			RAKE_test_out.append(rake_keywords)
			#NER_test_out.append(re.sub('[^A-Za-z]+',' ',named_entities).strip())
			#RAKE_test_out.append(re.sub('[^A-Za-z]+',' ',rake_keywords).strip())
	except:
		count_out=count_out-1			
	# if(count_out == 5):
	# 	break	

# =======================  Corpus Creation  =======================================


# ===================  Lexical Features and TF-IDF matrix   =======================
#train
vectorizer_semantic_text = CountVectorizer(token_pattern = u'(?ui)\\b\\w*[a-z]+\\w*\\b',stop_words = stopwords_ , decode_error = 'ignore',binary=True)
text_semantic = vectorizer_semantic_text.fit_transform(text_corpus)
text_semantic = text_semantic.todense()
text_semantic = text_semantic.tolist()

#test_in
text_semantic_test = vectorizer_semantic_text.transform(text_corpus_test)
text_semantic_test = text_semantic_test.todense()
text_semantic_test = text_semantic_test.tolist()

 #test_out
text_semantic_test_out = vectorizer_semantic_text.transform(text_corpus_test_out)
text_semantic_test_out = text_semantic_test_out.todense()
text_semantic_test_out = text_semantic_test_out.tolist()
print("==============  Lexical Start ==================")

#train
vectorizer_tfidf_text = TfidfVectorizer(token_pattern = u'(?ui)\\b\\w*[a-z]+\\w*\\b',stop_words = stopwords_ , decode_error = 'ignore',binary=True)
text_tfidf = vectorizer_tfidf_text.fit_transform(text_corpus)
text_tfidf = text_tfidf.todense()
text_tfidf = text_tfidf.tolist()
type(text_tfidf)
print(text_tfidf)
#pca(text_tfidf)
#test_in
text_tfidf_test = vectorizer_tfidf_text.transform(text_corpus_test)
text_tfidf_test = text_tfidf_test.todense()
text_tfidf_test = text_tfidf_test.tolist()

 #test_out
text_tfidf_test_out = vectorizer_tfidf_text.transform(text_corpus_test_out)
text_tfidf_test_out = text_tfidf_test_out.todense()
text_tfidf_test_out = text_tfidf_test_out.tolist()
#pca(text_tfidf_test_out)

print("==============  TF-IDF ==================")

# ===================  Lexical Features and TF-IDF matrix   =======================

# ===================  Bibliographic Features and Matrix   ========================

#train
vectorizer_tf_venue = TfidfVectorizer(token_pattern = u'(?ui)\\b\\w*[a-z]+\\w*\\b',stop_words = stopwords_ , decode_error = 'ignore',binary = True)
venue_tf = vectorizer_tf_venue.fit_transform(venue_corpus)
venue_tf = venue_tf.todense()
venue_tf = venue_tf.tolist()

#test_in
venue_tf_test = vectorizer_tf_venue.transform(venue_corpus_test)
venue_tf_test = venue_tf_test.todense()
venue_tf_test = venue_tf_test.tolist()

# #test_out
venue_tf_test_out = vectorizer_tf_venue.transform(venue_corpus_test_out)
venue_tf_test_out = venue_tf_test_out.todense()
venue_tf_test_out = venue_tf_test_out.tolist()




print("============== Venue =================")

#train
vectorizer_tf_title = TfidfVectorizer(token_pattern = u'(?ui)\\b\\w*[a-z]+\\w*\\b',stop_words = stopwords_ , decode_error = 'ignore',binary = True)
title_tf = vectorizer_tf_title.fit_transform(title_corpus)
title_tf = title_tf.todense()
title_tf = title_tf.tolist()


#test
title_tf_test = vectorizer_tf_title.transform(title_corpus_test)
title_tf_test = title_tf_test.todense()
title_tf_test = title_tf_test.tolist()

# #test_out
title_tf_test_out = vectorizer_tf_title.transform(title_corpus_test_out)
title_tf_test_out = title_tf_test_out.todense()
title_tf_test_out = title_tf_test_out.tolist()


print("============== Title =================")


##########################  Saving The Matrixes ####################
np.savetxt('Semantic_train.txt', text_semantic, delimiter='	',  newline='\n',fmt='%0.4f')
np.savetxt('Semantic_test.txt', text_semantic_test, delimiter='	',  newline='\n',fmt='%0.4f')
np.savetxt('Semantic_test_out.txt', text_semantic_test_out, delimiter='	',  newline='\n',fmt='%0.4f')

np.savetxt('Tfidf_train.txt', text_tfidf, delimiter='	',  newline='\n',fmt='%0.4f')
np.savetxt('Tfidf_test.txt', text_tfidf_test, delimiter='	',  newline='\n',fmt='%0.4f')
np.savetxt('Tfidf_test_out.txt', text_tfidf_test_out, delimiter='	',  newline='\n',fmt='%0.4f')
np.savetxt('Venue_train.txt', venue_tf, delimiter='	',  newline='\n',fmt='%0.4f')
np.savetxt('Venue_test.txt', venue_tf_test, delimiter='	',  newline='\n',fmt='%0.4f')
np.savetxt('Venue_test_out.txt', venue_tf_test_out, delimiter='	',  newline='\n',fmt='%0.4f')
np.savetxt('Title_train.txt', title_tf, delimiter='	',  newline='\n',fmt='%0.4f')
np.savetxt('Title_test.txt', title_tf_test, delimiter='	',  newline='\n',fmt='%0.4f')
np.savetxt('Title_test_out.txt', title_tf_test_out, delimiter='	',  newline='\n',fmt='%0.4f')
#np.savetxt('Semantic_train.json', RAKE, delimiter=',',  newline='\n',fmt="%s")
# np.savetxt('Semantic_test.json', RAKE_test, delimiter=',',  newline='\n',fmt="%s")

#with open("Semantic_train.txt","w") as fout:
#	for lis in RAKE:
#		fout.write(",".join(lis)+"\n")
#	for lis in NER:
#		fout.write(",".join(lis)+"\n")
#
#with open("Semantic_test.txt","w") as fout:
#	for lis in RAKE_test:
#		fout.write(",".join(lis)+"\n")
#	for lis in NER_test:
#		fout.write(",".join(lis)+"\n")	

#with open("Semantic_test_out.txt","w") as fout:
# 	for lis in RAKE_test_out:
#		fout.write(",".join(lis)+"\n")
#	for lis in NER_test_out:
#		fout.write(",".join(lis)+"\n")					


# ===================   Test  ========================
 
# print title_tf.shape,title_tf_vocab.shape
# print venue_tf.shape,venue_tf_vocab.shape
# print title_tf[0]
# print vectorizer_tf_title.get_feature_names()
# print "\n\n\n\n\n"
# print vectorizer_tf_title.vocabulary_ 
# print dict(zip(vectorizer.get_feature_names(),vectorizer.idf_))

# ===================   Test  ========================

# ===================  Bibliographic Features and Matrix   ========================

# ===================  Semantic Features and Matrix   =============================
dist=0.0
doc_sem = RAKE
temp=[]
semantic_dist = []
count = 0
semantic=[]
dist_semantic = []
t=[]
t=np.array(text_semantic)
t=t.T
t=np.cov(t)
for doc1 in range(len(text_semantic)):
	doc1_dist = []
	for doc2 in range(len(text_semantic)):
		#dist = float((spatial.distance.euclidean(text_tfidf[doc1],text_tfidf[doc2])))
		dist = cosin(text_semantic[doc1],text_semantic[doc2],t)
		# dist = 1.0 - dist
		doc1_dist.append(dist)

	dist_semantic.append(doc1_dist)
# ===================  Semantic Features and Matrix   =============================

# ===================  Distance Matrix   ==========================================

dist_tfidf = []
t=[]
t=np.array(text_tfidf)
t=t.T
t=np.cov(t)
for doc1 in range(len(text_tfidf)):
	doc1_dist = []
	for doc2 in range(len(text_tfidf)):
		#dist = float((spatial.distance.euclidean(text_tfidf[doc1],text_tfidf[doc2])))
		dist = cosin(text_tfidf[doc1],text_tfidf[doc2],t)
		# dist = 1.0 - dist
		doc1_dist.append(dist)

	dist_tfidf.append(doc1_dist)

dist_venue = []
t=[]
t=np.array(venue_tf)
t=t.T
t=np.cov(t)
print("Dist_tfidf")
for doc1 in range(len(venue_tf)):
	doc1_dist = []
	for doc2 in range(len(venue_tf)):
		#dist = float((spatial.distance.euclidean(venue_tf[doc1],venue_tf[doc2])))
		dist = cosin(venue_tf[doc1],venue_tf[doc2],t)
		doc1_dist.append(dist)

	dist_venue.append(doc1_dist)

# dist_venue = [1 if math.isnan(x) else x for lis in dist_venue for x in lis]

print("Dist_venue")

dist_title = []
t=[]
t=np.array(title_tf)
t=t.T
t=np.cov(t)
for doc1 in range(len(title_tf)):
	doc1_dist = []
	for doc2 in range(len(title_tf)):
		#dist = float((spatial.distance.euclidean(title_tf[doc1],title_tf[doc2])))
		dist = cosin(title_tf[doc1],title_tf[doc2],t)
		doc1_dist.append(dist)

	dist_title.append(doc1_dist)

print("Dist_title")

# dist_title = [1 if math.isnan(x) else x for lis in dist_title for x in lis]

np.savetxt('Dist_tfidf.txt', dist_tfidf, delimiter='	',  newline='\n',fmt='%0.4f')
np.savetxt('Dist_venue.txt', dist_venue, delimiter='	',  newline='\n',fmt='%0.4f')
# np.savetxt('Dist_venue_vocab.txt', dist_venue_vocab, delimiter='	',newline='\n',fmt='%0.4f')
np.savetxt('Dist_title.txt', dist_title, delimiter='	',  newline='\n',fmt='%0.4f')
# np.savetxt('Dist_title_vocab.txt', dist_title_vocab, delimiter='	',  newline='\n',fmt='%0.4f')
# np.savetxt('Dist_author.txt', dist_author, delimiter='	',  newline='\n',fmt='%0.4f')
np.savetxt('Dist_semantic.txt', dist_semantic, delimiter='	',  newline='\n',fmt='%0.4f')


print("Count Inscope : ",count)
# print "Count Outscope : ",count_out
		
# ===================  Distance Matrix   ==========================================

# ===================  Multi Objective Optimization   ============================= 



# ===================  Multi Objective Optimization   ============================= 
