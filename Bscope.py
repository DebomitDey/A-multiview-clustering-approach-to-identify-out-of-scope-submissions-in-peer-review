#for inscope its 0
#for outscope its 1

import os
import math
from scipy import spatial
from collections import Counter  
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from numpy import linalg as LA
from operator import sub,mul
from scipy.spatial import distance
alpha=0.001
def cosin(temp1,temp2,temp):
	dist=0.0
	#dist = math.fabs(np.dot(temp1,temp2)/(alpha+math.sqrt(LA.norm(temp1)*LA.norm(temp2))))
	#dist=math.fabs(1.00-dist)
	#temp1=list(map(sub,temp1,temp2))
	#dist=math.sqrt(sum(list(map(mul,temp1,temp1))))
	dist=distance.mahalanobis(temp1,temp2,temp)
	return dist
tfidf_train = []
venue_train = []
title_train = []
semantic_train = []
base = "/home/debomit/Desktop/hh/"
#model =  KeyedVectors.load_word2vec_format(base+"pubmed-and-PMC-w2wikipediav.bin",unicode_errors= 'ignore',binary=True)
tfidf_test = []
venue_test = []
title_test = []
semantic_test = []

tfidf_test_out = []
venue_test_out = []
title_test_out = []
semantic_test_out = []

dist_tfidf = []
dist_venue = []
dist_title = []
dist_semantic = []

mem_consensus = []
mem_first = []
mem_second = []
mem_third = []
mem_fourth = []
print("hello")

########################  Train Matrices  ########################################

with open("Tfidf_train.txt","r") as fin:
	tfidf = fin.readlines()
	for lines in tfidf:
		lines.replace("\n",'')
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split("\t"))
		member_cluster=list(map(float,member_cluster))
		tfidf_train.append(member_cluster)
print("hello")
with open("Title_train.txt","r") as fin:
	title = fin.readlines()
	for lines in title:
		lines.replace("\n",'')
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split("\t"))
		member_cluster=list(map(float,member_cluster))
		title_train.append(member_cluster)		
print("hello")
with open("Venue_train.txt","r") as fin:
	venue = fin.readlines()
	for lines in venue:
		lines.replace("\n",'')
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split("\t"))
		member_cluster=list(map(float,member_cluster))
		venue_train.append(member_cluster)
print("hello")
with open("Semantic_train.txt","r") as fin:
	venue = fin.readlines()
	for lines in venue:
		lines.replace("\n",'')
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split("\t"))
		member_cluster=list(map(float,member_cluster))
		semantic_train.append(member_cluster)	
print("hello")
print("hello")
########################  Train Matrices  ########################################


########################  Test Matrices  ########################################

with open("Tfidf_test.txt","r") as fin:
	tfidf = fin.readlines()
	for lines in tfidf:
		lines.replace("\n",'')
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split("\t"))
		member_cluster=list(map(float,member_cluster))
		tfidf_test.append(member_cluster)
print("hello")
with open("Title_test.txt","r") as fin:
	title = fin.readlines()
	for lines in title:
		lines.replace("\n",'')
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split("\t"))
		member_cluster=list(map(float,member_cluster))
		title_test.append(member_cluster)		

with open("Venue_test.txt","r") as fin:
	venue = fin.readlines()
	for lines in venue:
		lines.replace("\n",'')
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split("\t"))
		member_cluster=list(map(float,member_cluster))
		venue_test.append(member_cluster)
print("hello")
with open("Semantic_test.txt","r") as fin:
	venue = fin.readlines()
	for lines in venue:
		lines.replace("\n",'')
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split("\t"))
		member_cluster=list(map(float,member_cluster))
		semantic_test.append(member_cluster)	
print("hello")
with open("Tfidf_test_out.txt","r") as fin:
	tfidf = fin.readlines()
	for lines in tfidf:
		lines.replace("\n",'')
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split("\t"))
		member_cluster=list(map(float,member_cluster))
		tfidf_test_out.append(member_cluster)
print("hello")
with open("Title_test_out.txt","r") as fin:
	title = fin.readlines()
	for lines in title:
		lines.replace("\n",'')
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split("\t"))
		member_cluster=list(map(float,member_cluster))
		title_test_out.append(member_cluster)		
print("hello")
with open("Venue_test_out.txt","r") as fin:
	venue = fin.readlines()
	for lines in venue:
		lines.replace("\n",'')
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split("\t"))
		member_cluster=list(map(float,member_cluster))
		venue_test_out.append(member_cluster)
print("hello")
with open("Semantic_test_out.txt","r") as fin:
	venue = fin.readlines()
	for lines in venue:
		lines.replace("\n",'')
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split("\t"))
		member_cluster=list(map(float,member_cluster))
		semantic_test_out.append(member_cluster)			
print("hello")
print("hello")
########################  Test Matrices  ########################################

semantic_cov=[]
title_cov=[]
tfidf_cov=[]
venue_cov=[]
#######################  Distance Matrices ######################################

with open("Dist_tfidf.txt","r") as fin:
	tfidf = fin.readlines()
	for lines in tfidf:
		lines.replace("\n",'')
		#print(lines)
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split("\t"))
		member_cluster=list(map(float,member_cluster))
		dist_tfidf.append(member_cluster)

with open("Dist_title.txt","r") as fin:
	title = fin.readlines()
	for lines in title:
		lines.replace("\n",'')
		#print(lines)
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split("\t"))
		member_cluster=list(map(float,member_cluster))
		dist_title.append(member_cluster)		

with open("Dist_venue.txt","r") as fin:
	venue = fin.readlines()
	for lines in venue:
		lines.replace("\n",'')
		#print(lines)
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split("\t"))
		member_cluster=list(map(float,member_cluster))
		dist_venue.append(member_cluster)

with open("Dist_semantic.txt","r") as fin:
	semantic = fin.readlines()
	for lines in semantic:
		lines.replace("\n",'')
		#print(lines)
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split("\t"))
		member_cluster=list(map(float,member_cluster))
		dist_semantic.append(member_cluster)				
#print(dist_semantic)
semantic_cov=np.array(dist_semantic)
tfidf_cov=np.array(dist_tfidf)
venue_cov=np.array(dist_venue)
title_cov=np.array(dist_title)

semantic_cov=semantic_cov.T
tfidf_cov=tfidf_cov.T
venue_cov=venue_cov.T
title_cov=title_cov.T

semantic_cov=np.cov(semantic_cov)
tfidf_cov=np.cov(tfidf_cov)
venue_cov=np.cov(venue_cov)
title_cov=np.cov(title_cov)
#######################  Distance Matrices ######################################

###################### Membeship Matrices #######################################

with open("membership_consensus1.txt","r") as fin:
	member = fin.readlines()
	for lines in member:
		lines.replace("\n",'')
		#print(lines)
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split("\t"))
		member_cluster=list(map(float,member_cluster))
		mem_consensus.append(member_cluster)		
for i in mem_consensus:
	print(type(i))
with open("Membership1.txt","r") as fin:
	member = fin.readlines()
	for l in range(len(member)):
		lines=member[l].replace("\n",'')
		lines.replace('1.000000','1')
		lines.replace('0.000000','0')
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split(" "))
		member_cluster=member_cluster[:-1]
		member_cluster=list(map(float,member_cluster))
		mem_first.append(member_cluster)

with open("Membership2.txt","r") as fin:
	member = fin.readlines()
	for l in range(len(member)):
		lines=member[l].replace("\n",'')
		lines.replace('1.000000','1')
		lines.replace('0.000000','0')
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split(" "))
		member_cluster=member_cluster[:-1]
		member_cluster=list(map(float,member_cluster))
		mem_second.append(member_cluster)

with open("Membership3.txt","r") as fin:
	member = fin.readlines()
	for l in range(len(member)):
		lines=member[l].replace("\n",'')
		lines.replace('1.000000','1')
		lines.replace('0.000000','0')
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split(" "))
		member_cluster=member_cluster[:-1]
		member_cluster=list(map(float,member_cluster))
		mem_third.append(member_cluster)

with open("Membership4.txt","r") as fin:
	member = fin.readlines()
	for l in range(len(member)):
		lines=member[l].replace("\n",'')
		lines.replace('1.000000','1')
		lines.replace('0.000000','0')
		member_cluster = lines.split('\n')
		member_cluster=list(str(member_cluster[0]).split(" "))
		member_cluster=member_cluster[:-1]
		member_cluster=list(map(float,member_cluster))
		mem_fourth.append(member_cluster)		


###################### Membeship Matrices #######################################
mem_cons=[]
mem1_clus = []
mem2_clus = []
mem3_clus = []
mem4_clus = []
for cluster in mem_consensus:
	mem_clus = []
	for i,member in enumerate(cluster):
		if(member>0):
			mem_clus.append(i)
	print(mem_clus)
	if(mem_clus!=[]):	
		mem_cons.append(mem_clus)
print("mem_cons")
for c in mem_cons:
	print(c)
	print(type(c))
for cluster in mem_first:
	mem_clus = []
	for i,member in enumerate(cluster):
		#print float(round(member,1))
		if(member>0):
			mem_clus.append(i)
	if(mem_clus!=[]):	
		mem1_clus.append(mem_clus)
print("mem1_clus")
print(len(mem1_clus))

for cluster in mem_second:
	mem_clus = []
	for i,member in enumerate(cluster):
		if(member>0):
			mem_clus.append(i)
	if(mem_clus!=[]):	
		mem2_clus.append(mem_clus)
print("mem2_clus")
print(len(mem2_clus))
for cluster in mem_third:
	mem_clus = []
	for i,member in enumerate(cluster):
		if(member>0):
			mem_clus.append(i)
	if(mem_clus!=[]):	
		mem3_clus.append(mem_clus)
print("mem3_clus")
print(len(mem3_clus))

for cluster in mem_fourth:
	mem_clus = []
	for i,member in enumerate(cluster):
		if(member>0):
			mem_clus.append(i)
	if(mem_clus!=[]):
		mem4_clus.append(mem_clus)	
print("mem4_clus")
print(len(mem4_clus))

cluscentre_view1 = []
cluscentre_view2 = []
cluscentre_view3 = []
cluscentre_view4 = []	
dist=0.0
dist_from_cent1=[]
dist_from_cent2=[]
dist_from_cent3=[]
dist_from_cent4=[]
conscluscentre_view1 = []
conscluscentre_view2 = []
conscluscentre_view3 = []
conscluscentre_view4 = []	

consdist_from_cent1=[]
consdist_from_cent2=[]
consdist_from_cent3=[]
consdist_from_cent4=[]

avg_dist_from_cent1=[]
avg_dist_from_cent2=[]
avg_dist_from_cent3=[]
avg_dist_from_cent4=[]

avg_consdist_from_cent1=[]
avg_consdist_from_cent2=[]
avg_consdist_from_cent3=[]
avg_consdist_from_cent4=[]
#--------------------------------------------------------------semantic cluscentre_view1--------------------------------------------------
for i,cluster in enumerate(mem1_clus):
	dist_mem = []
	if(cluster!=[]):
		for mem_1 in cluster:
			dist = 0.0
			for mem_2 in cluster:
				dist = dist + float(dist_semantic[mem_1][mem_2])					
			dist = float(dist/len(cluster))	
			dist_mem.append(dist)
		ind_clus = dist_mem.index(min(dist_mem))
		cluscentre_view1.append(mem1_clus[i][ind_clus])
#-----------------------------------------------------------------semantic consensus cluscentre_view1----------------------------------
for i,cluster in enumerate(mem_cons):
	dist_mem = []
	if(cluster!=[]):
		for mem_1 in cluster:
			dist = 0.0
			for mem_2 in cluster:
				dist = dist + float(dist_semantic[mem_1][mem_2])					
			dist = float(dist/len(cluster))	
			dist_mem.append(dist)
		ind_clus = dist_mem.index(min(dist_mem))
		conscluscentre_view1.append(mem_cons[i][ind_clus])
print("cluscentre_view1")
print(cluscentre_view1)
#------------------------------------------------------------------semantic dist_from_cent1----------------------------------------
for i,cluster in enumerate(mem1_clus):
	centre=cluscentre_view1[i]
	dist_mem=[]
	for point in cluster:
		dist_mem.append(float(dist_semantic[centre][point]))
	dist_from_cent1.append(max(dist_mem))
	avg_dist_from_cent1.append(float(sum(dist_mem)/(len(dist_mem))))
#------------------------------------------------------------------semantic consensus dist_from_cent1-----------------------------
for i,cluster in enumerate(mem_cons):
	centre=conscluscentre_view1[i]
	dist_mem=[]
	for point in cluster:
		dist_mem.append(float(dist_semantic[centre][point]))
	consdist_from_cent1.append(max(dist_mem))
	avg_consdist_from_cent1.append(float(sum(dist_mem)/(len(dist_mem))))
#-----------------------------------------------------------------tfidf cluscentre_view2----------------------------------
for i,cluster in enumerate(mem2_clus):
	dist_mem = []
	if(cluster!=[]):
		for mem_1 in cluster:
			dist = 0.0
			for mem_2 in cluster:
				dist = dist + float(dist_tfidf[mem_1][mem_2])					
			dist = float(dist/len(cluster))	
			dist_mem.append(dist)
		ind_clus = dist_mem.index(min(dist_mem))
		cluscentre_view2.append(mem2_clus[i][ind_clus])
print("cluscentre_view2")
print(cluscentre_view2)
#------------------------------------------------------------------tfidf consensus cluscentre_view2---------------------------------
for i,cluster in enumerate(mem_cons):
	dist_mem = []
	if(cluster!=[]):
		for mem_1 in cluster:
			dist = 0.0
			for mem_2 in cluster:
				dist = dist + float(dist_tfidf[mem_1][mem_2])					
			dist = float(dist/len(cluster))	
			dist_mem.append(dist)
		ind_clus = dist_mem.index(min(dist_mem))
		conscluscentre_view2.append(mem_cons[i][ind_clus])
#-----------------------------------------------------------------tfidf dist_from_cent2------------------------------------
for i,cluster in enumerate(mem2_clus):
	centre=cluscentre_view2[i]
	dist_mem=[]
	for point in cluster:
		dist_mem.append(float(dist_tfidf[centre][point]))
	dist_from_cent2.append(max(dist_mem))
	avg_dist_from_cent2.append(float(sum(dist_mem)/(len(dist_mem))))
#-------------------------------------------------------------------tfidf consensus dist_from_cent2------------------------------
for i,cluster in enumerate(mem_cons):
	centre=conscluscentre_view2[i]
	dist_mem=[]
	for point in cluster:
		dist_mem.append(float(dist_tfidf[centre][point]))
	consdist_from_cent2.append(max(dist_mem))
	avg_consdist_from_cent2.append(float(sum(dist_mem)/(len(dist_mem))))


#----------------------------------------------------------------title cluscentre_view3-------------------------------------
for i,cluster in enumerate(mem3_clus):
	dist_mem = []
	if(cluster!=[]):
		for mem_1 in cluster:
			dist = 0.0
			for mem_2 in cluster:
				dist = dist + float(dist_title[mem_1][mem_2])					
			dist = float(dist/len(cluster))	
			dist_mem.append(dist)
		ind_clus = dist_mem.index(min(dist_mem))
		cluscentre_view3.append(mem3_clus[i][ind_clus])
print("cluscentre_view3")
print(cluscentre_view3)
#-------------------------------------------------------------title consensus cluscentre_view3------------------------------------
for i,cluster in enumerate(mem_cons):
	dist_mem = []
	if(cluster!=[]):
		for mem_1 in cluster:
			dist = 0.0
			for mem_2 in cluster:
				dist = dist + float(dist_title[mem_1][mem_2])					
			dist = float(dist/len(cluster))	
			dist_mem.append(dist)
		ind_clus = dist_mem.index(min(dist_mem))
		conscluscentre_view3.append(mem_cons[i][ind_clus])
#--------------------------------------------------------------title dist_from_cent3-----------------------------------------------
for i,cluster in enumerate(mem3_clus):
	centre=cluscentre_view3[i]
	dist_mem=[]
	for point in cluster:
		dist_mem.append(float(dist_title[centre][point]))
	dist_from_cent3.append(max(dist_mem))
	avg_dist_from_cent3.append(float(sum(dist_mem)/(len(dist_mem))))
#--------------------------------------------------------------title consensus dist_from_cent3---------------------------------
for i,cluster in enumerate(mem_cons):
	centre=conscluscentre_view3[i]
	dist_mem=[]
	for point in cluster:
		dist_mem.append(float(dist_title[centre][point]))
	consdist_from_cent3.append(max(dist_mem))
	avg_consdist_from_cent3.append(float(sum(dist_mem)/(len(dist_mem))))

#333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333






#-----------------------------------------------------venue cluscentre_view4-------------------------------------------
for i,cluster in enumerate(mem4_clus):
	dist_mem = []
	if(cluster!=[]):
		for mem_1 in cluster:
			dist = 0.0
			for mem_2 in cluster:
				dist = dist + float(dist_venue[mem_1][mem_2])					
			dist = float(dist/len(cluster))	
			dist_mem.append(dist)
		ind_clus = dist_mem.index(min(dist_mem))
		cluscentre_view4.append(mem4_clus[i][ind_clus])
print("cluscentre_view4")
print(cluscentre_view4)
#-----------------------------------------------------venue consensus cluscentre_view4------------------------------------
for i,cluster in enumerate(mem_cons):
	dist_mem = []
	if(cluster!=[]):
		for mem_1 in cluster:
			dist = 0.0
			for mem_2 in cluster:
				dist = dist + float(dist_venue[mem_1][mem_2])					
			dist = float(dist/len(cluster))	
			dist_mem.append(dist)
		ind_clus = dist_mem.index(min(dist_mem))
		conscluscentre_view4.append(mem_cons[i][ind_clus])
#-----------------------------------------------------venue dist_from_cent4-----------------------------------------------
for i,cluster in enumerate(mem4_clus):
	centre=cluscentre_view4[i]
	dist_mem=[]
	for point in cluster:
		dist_mem.append(float(dist_venue[centre][point]))
	dist_from_cent4.append(max(dist_mem))
	avg_dist_from_cent4.append(float(sum(dist_mem)/(len(dist_mem))))
#------------------------------------------------------venue consensus dist_from_cent4-------------------------------------
for i,cluster in enumerate(mem_cons):
	centre=conscluscentre_view4[i]
	dist_mem=[]
	for point in cluster:
		dist_mem.append(float(dist_venue[centre][point]))
	consdist_from_cent4.append(max(dist_mem))
	avg_consdist_from_cent4.append(float(sum(dist_mem)/(len(dist_mem))))
#print("hello")
############ Cluster belonging for test set ##############


#############################################################  normal 4 view semantic tfidf title venue  ############################# 
inscope_view1=[]
inscope_view2=[]
inscope_view3=[]
inscope_view4=[]

outscope_view1=[]
outscope_view2=[]
outscope_view3=[]
outscope_view4=[]
clus_belong_view1=[]
clus_belong_view2=[]
clus_belong_view3=[]
clus_belong_view4=[]
clus_belong_view1_out=[]
clus_belong_view2_out=[]
clus_belong_view3_out=[]
clus_belong_view4_out=[]

covr=[]
covr=np.array(semantic_train)
covr=covr.T
covr=np.cov(covr)
for docs in semantic_test:
	dist_clus = []
	for cluster in cluscentre_view1:
		docs=list(map(float,docs))
		temp=[]
		temp=list(map(float,semantic_train[cluster]))
		dist=cosin(docs,temp,covr)
		#dist = float(spatial.distance.cosine(docs,temp))
		#dist=model.wmdistance(docs,semantic_train[cluster])
		dist_clus.append(dist)
	clus_no = dist_clus.index(min(dist_clus))	
	clus_belong_view1.append(clus_no)
covr=[]
covr=np.array(semantic_train)
covr=covr.T
covr=np.cov(covr)
for docs in semantic_test_out:
	dist_clus = []
	for cluster in cluscentre_view1:
		docs=list(map(float,docs))
		temp=[]
		temp=list(map(float,semantic_train[cluster]))
		dist=cosin(docs,temp,covr)
		#dist = float(spatial.distance.cosine(docs,temp))
		#dist=model.wmdistance(docs,semantic_train[cluster])
		dist_clus.append(dist)
	clus_no = dist_clus.index(min(dist_clus))	
	clus_belong_view1_out.append(clus_no)
covr=[]
covr=np.array(tfidf_train)
covr=covr.T
covr=np.cov(covr)
for docs in tfidf_test:
	dist_clus = []
	for cluster in cluscentre_view2:
		docs=list(map(float,docs))
		temp=[]
		temp=list(map(float,tfidf_train[cluster]))
		#dist = float(spatial.distance.euclidean(docs,temp))
		dist=cosin(docs,temp,covr)
		dist_clus.append(dist)
	clus_no = dist_clus.index(min(dist_clus))
	clus_belong_view2.append(clus_no)
covr=[]
covr=np.array(tfidf_train)
covr=covr.T
covr=np.cov(covr)
for docs in tfidf_test_out:
	dist_clus = []
	for cluster in cluscentre_view2:
		docs=list(map(float,docs))
		temp=[]
		temp=list(map(float,tfidf_train[cluster]))
		#dist = float(spatial.distance.euclidean(docs,temp))
		dist=cosin(docs,temp,covr)
		dist_clus.append(dist)
	clus_no = dist_clus.index(min(dist_clus))
	clus_belong_view2_out.append(clus_no)
covr=[]
covr=np.array(title_train)
covr=covr.T
covr=np.cov(covr)
for docs in title_test:
	dist_clus = []
	for cluster in cluscentre_view3:
		docs=list(map(float,docs))
		temp=[]
		temp=list(map(float,title_train[cluster]))
		#dist = float(spatial.distance.euclidean(docs,temp))
		dist=cosin(docs,temp,covr)
		dist_clus.append(dist)
	clus_no = dist_clus.index(min(dist_clus))	
	clus_belong_view3.append(clus_no)
covr=[]
covr=np.array(title_train)
covr=covr.T
covr=np.cov(covr)
for docs in title_test_out:
	dist_clus = []
	for cluster in cluscentre_view3:
		docs=list(map(float,docs))
		temp=[]
		temp=list(map(float,title_train[cluster]))
		#dist = float(spatial.distance.euclidean(docs,temp))
		dist=cosin(docs,temp,covr)
		dist_clus.append(dist)
	clus_no = dist_clus.index(min(dist_clus))	
	clus_belong_view3_out.append(clus_no)

covr=[]
covr=np.array(venue_train)
covr=covr.T
covr=np.cov(covr)
for docs in venue_test:
	dist_clus = []
	for cluster in cluscentre_view4:
		docs=list(map(float,docs))
		temp=[]
		temp=list(map(float,venue_train[cluster]))
		#dist = float(spatial.distance.euclidean(docs,temp))
		dist=cosin(docs,temp,covr)
		dist_clus.append(dist)
	clus_no = dist_clus.index(min(dist_clus))	
	clus_belong_view4.append(clus_no)

covr=[]
covr=np.array(venue_train)
covr=covr.T
covr=np.cov(covr)
for docs in venue_test_out:
	dist_clus = []
	for cluster in cluscentre_view4:
		docs=list(map(float,docs))
		temp=[]
		temp=list(map(float,venue_train[cluster]))
		#dist = float(spatial.distance.euclidean(docs,temp))
		dist=cosin(docs,temp,covr)
		dist_clus.append(dist)
	clus_no = dist_clus.index(min(dist_clus))	
	clus_belong_view4_out.append(clus_no)

#-----------------------------------------------------------semantic-----------------------------------------------------
with open("resultdebo1.txt","a") as fout:
	fout.write("semantic inscope\n")
covr=[]
covr=np.array(semantic_train)
covr=covr.T
covr=np.cov(covr)
for i,docs in enumerate(semantic_test):
	docs=list(map(float,docs))
	temp=[]
	temp=list(map(float,semantic_train[cluscentre_view1[clus_belong_view1[i]]]))
	dist=cosin(docs,temp,covr)
	#temp=str(semantic_train[cluscentre_view1[clus_belong_view1[i]]])
	#dist = model.wmdistance(docs,temp)
	if(dist<(dist_from_cent1[clus_belong_view1[i]])):
		inscope_view1.append(1)
	else:
		inscope_view1.append(0)
	with open("resultdebo1.txt","a") as fout:	
		fout.write(str(dist)+"\n")
with open("resultdebo1.txt","a") as fout:
	fout.write("semantic inscope end\n")
#-avg_dist_from_cent1[clus_belong_view1[i]]+
with open("resultdebo1.txt","a") as fout:
	fout.write("semantic outscope\n")
covr=[]
covr=np.array(semantic_train)
covr=covr.T
covr=np.cov(covr)
for i,docs in enumerate(semantic_test_out):
	docs=list(map(float,docs))
	temp=[]
	temp=list(map(float,semantic_train[cluscentre_view1[clus_belong_view1_out[i]]]))
	#temp=str(semantic_train[cluscentre_view1[clus_belong_view1_out[i]]])
	#dist = model.wmdistance(docs,temp)
	dist=cosin(docs,temp,covr)
	if(dist<(dist_from_cent1[clus_belong_view1_out[i]])):
		outscope_view1.append(1)
	else:
		outscope_view1.append(0)
	with open("resultdebo1.txt","a") as fout:
		fout.write(str(dist)+"\n")
with open("resultdebo1.txt","a") as fout:
	fout.write("semantic outscope end\n")
#-avg_dist_from_cent1[clus_belong_view1_out[i]]+

print("inscope_view1")
print(inscope_view1)
print("outscope_view1")
print(outscope_view1)
#------------------------------------------------------------tfidf-------------------------------------------------------
with open("resultdebo1.txt","a") as fout:
	fout.write("tfidf inscope\n")
covr=[]
covr=np.array(tfidf_train)
covr=covr.T
covr=np.cov(covr)
for i,docs in enumerate(tfidf_test):
	docs=list(map(float,docs))
	temp=[]
	temp=list(map(float,tfidf_train[cluscentre_view2[clus_belong_view2[i]]]))
	#dist = float(spatial.distance.euclidean(docs,temp))
	dist=cosin(docs,temp,covr)
	if(dist!=1.0 and dist<(dist_from_cent2[clus_belong_view2[i]])):
		inscope_view2.append(1)
	else:
		inscope_view2.append(0)
	with open("resultdebo1.txt","a") as fout:	
		fout.write(str(dist)+"\n")
with open("resultdebo1.txt","a") as fout:
	fout.write("tfidf inscope end\n")
#-avg_dist_from_cent2[clus_belong_view2[i]]+
with open("resultdebo1.txt","a") as fout:
	fout.write("tfidf outscope\n")
covr=[]
covr=np.array(tfidf_train)
covr=covr.T
covr=np.cov(covr)
for i,docs in enumerate(tfidf_test_out):
	docs=list(map(float,docs))
	temp=[]
	temp=list(map(float,tfidf_train[cluscentre_view2[clus_belong_view2_out[i]]]))
	#dist = float(spatial.distance.euclidean(docs,temp))
	dist=cosin(docs,temp,covr)
	if(dist!=1.0 and dist<(dist_from_cent2[clus_belong_view2_out[i]])):
		outscope_view2.append(1)
	else:
		outscope_view2.append(0)
	with open("resultdebo1.txt","a") as fout:
		fout.write(str(dist)+"\n")
with open("resultdebo1.txt","a") as fout:
	fout.write("tfidf outscope end\n")
#-avg_dist_from_cent2[clus_belong_view2_out[i]]+
print("inscope_view2")
print(inscope_view2)
print("outscope_view2")
print(outscope_view2)

#------------------------------------------------------------title------------------------------------------------------
with open("resultdebo1.txt","a") as fout:
	fout.write("title inscope\n")
covr=[]
covr=np.array(title_train)
covr=covr.T
covr=np.cov(covr)
for i,docs in enumerate(title_test):
	docs=list(map(float,docs))
	temp=[]
	temp=list(map(float,title_train[cluscentre_view3[clus_belong_view3[i]]]))
	#dist = float(spatial.distance.euclidean(docs,temp))
	dist=cosin(docs,temp,covr)
	if(dist!=1.0 and dist<(dist_from_cent3[clus_belong_view3[i]])):
		inscope_view3.append(1)
	else:
		inscope_view3.append(0)
	with open("resultdebo1.txt","a") as fout:	
		fout.write(str(dist)+"\n")
with open("resultdebo1.txt","a") as fout:
	fout.write("title inscope end\n")
#-avg_dist_from_cent3[clus_belong_view3[i]]+
with open("resultdebo1.txt","a") as fout:
	fout.write("title outscope\n")
covr=[]
covr=np.array(title_train)
covr=covr.T
covr=np.cov(covr)
for i,docs in enumerate(title_test_out):
	docs=list(map(float,docs))
	temp=[]
	temp=list(map(float,title_train[cluscentre_view3[clus_belong_view3_out[i]]]))
	#dist = float(spatial.distance.euclidean(docs,temp))
	dist=cosin(docs,temp,covr)
	if(dist!=1.0 and dist<(dist_from_cent3[clus_belong_view3_out[i]])):
		outscope_view3.append(1)
	else:
		outscope_view3.append(0)
	with open("resultdebo1.txt","a") as fout:
		fout.write(str(dist)+"\n")
with open("resultdebo1.txt","a") as fout:
	fout.write("title outscope end\n")
#-avg_dist_from_cent3[clus_belong_view3_out[i]]+
print("inscope_view3")
print(inscope_view3)
print("outscope_view3")
print(outscope_view3)

#------------------------------------------------------------venue---------------------------------------------------------
with open("resultdebo1.txt","a") as fout:
	fout.write("venue inscope\n")
covr=[]
covr=np.array(venue_train)
covr=covr.T
covr=np.cov(covr)
for i,docs in enumerate(venue_test):
	docs=list(map(float,docs))
	temp=[]
	temp=list(map(float,venue_train[cluscentre_view4[clus_belong_view4[i]]]))
	#dist = float(spatial.distance.euclidean(docs,temp))
	dist=cosin(docs,temp,covr)
	if(dist!=1.0 and dist<(dist_from_cent4[clus_belong_view4[i]])):
		inscope_view4.append(1)
	else:
		inscope_view4.append(0)
	with open("resultdebo1.txt","a") as fout:	
		fout.write(str(dist)+"\n")
with open("resultdebo1.txt","a") as fout:
	fout.write("venue inscope end\n")
#-avg_dist_from_cent4[clus_belong_view4[i]]+
with open("resultdebo1.txt","a") as fout:
	fout.write("venue outscope\n")
covr=[]
covr=np.array(venue_train)
covr=covr.T
covr=np.cov(covr)
for i,docs in enumerate(venue_test_out):
	docs=list(map(float,docs))
	temp=[]
	temp=list(map(float,venue_train[cluscentre_view4[clus_belong_view4_out[i]]]))
	#dist = float(spatial.distance.euclidean(docs,temp))
	dist=cosin(docs,temp,covr)
	if(dist!=1.0 and dist<(dist_from_cent4[clus_belong_view4_out[i]])):
		outscope_view4.append(1)
	else:
		outscope_view4.append(0)
	with open("resultdebo1.txt","a") as fout:
		fout.write(str(dist)+"\n")
#-avg_dist_from_cent4[clus_belong_view4_out[i]]+
with open("resultdebo1.txt","a") as fout:
	fout.write("venue outscope end\n")
print("inscope_view4")
print(inscope_view4)
print("outscope_view4")
print(outscope_view4)




#print(inscope_view1)
#print(inscope_view2)
#print(inscope_view3)
#print(inscope_view4)
#print(outscope_view1)
#print(outscope_view2)
#print(outscope_view3)
#print(outscope_view4)



######################################## consensus view #########################################################

Cinscope_view1=[]
Cinscope_view2=[]
Cinscope_view3=[]
Cinscope_view4=[]

Coutscope_view1=[]
Coutscope_view2=[]
Coutscope_view3=[]
Coutscope_view4=[]
Cclus_belong_view1=[]
Cclus_belong_view2=[]
Cclus_belong_view3=[]
Cclus_belong_view4=[]
Cclus_belong_view1_out=[]
Cclus_belong_view2_out=[]
Cclus_belong_view3_out=[]
Cclus_belong_view4_out=[]
covr=[]
covr=np.array(semantic_train)
covr=covr.T
covr=np.cov(covr)
for docs in semantic_test:
	dist_clus = []
	for cluster in conscluscentre_view1:
		docs=list(map(float,docs))
		temp=[]
		temp=list(map(float,semantic_train[cluster]))
		#dist = float(spatial.distance.cosine(docs,temp))
		#dist=model.wmdistance(docs,semantic_train[cluster])
		dist=cosin(docs,temp,covr)
		dist_clus.append(dist)
	clus_no = dist_clus.index(min(dist_clus))	
	Cclus_belong_view1.append(clus_no)
covr=[]
covr=np.array(semantic_train)
covr=covr.T
covr=np.cov(covr)
for docs in semantic_test_out:
	dist_clus = []
	for cluster in conscluscentre_view1:
		docs=list(map(float,docs))
		temp=[]
		temp=list(map(float,semantic_train[cluster]))
		#dist = float(spatial.distance.cosine(docs,temp))
		#dist=model.wmdistance(docs,semantic_train[cluster])
		dist=cosin(docs,temp,covr)
		dist_clus.append(dist)
	clus_no = dist_clus.index(min(dist_clus))	
	Cclus_belong_view1_out.append(clus_no)

covr=[]
covr=np.array(tfidf_train)
covr=covr.T
covr=np.cov(covr)
for docs in tfidf_test:
	dist_clus = []
	for cluster in conscluscentre_view2:
		docs=list(map(float,docs))
		temp=[]
		temp=list(map(float,tfidf_train[cluster]))
		#dist = float(spatial.distance.euclidean(docs,temp))
		dist=cosin(docs,temp,covr)
		dist_clus.append(dist)
	clus_no = dist_clus.index(min(dist_clus))
	Cclus_belong_view2.append(clus_no)
covr=[]
covr=np.array(tfidf_train)
covr=covr.T
covr=np.cov(covr)
for docs in tfidf_test_out:
	dist_clus = []
	for cluster in conscluscentre_view2:
		docs=list(map(float,docs))
		temp=[]
		temp=list(map(float,tfidf_train[cluster]))
		#dist = float(spatial.distance.euclidean(docs,temp))
		dist=cosin(docs,temp,covr)
		dist_clus.append(dist)
	clus_no = dist_clus.index(min(dist_clus))
	Cclus_belong_view2_out.append(clus_no)

covr=[]
covr=np.array(title_train)
covr=covr.T
covr=np.cov(covr)
for docs in title_test:
	dist_clus = []
	for cluster in conscluscentre_view3:
		docs=list(map(float,docs))
		temp=[]
		temp=list(map(float,title_train[cluster]))
		#dist = float(spatial.distance.euclidean(docs,temp))
		dist=cosin(docs,temp,covr)
		dist_clus.append(dist)
	clus_no = dist_clus.index(min(dist_clus))	
	Cclus_belong_view3.append(clus_no)

covr=[]
covr=np.array(title_train)
covr=covr.T
covr=np.cov(covr)
for docs in title_test_out:
	dist_clus = []
	for cluster in conscluscentre_view3:
		docs=list(map(float,docs))
		temp=[]
		temp=list(map(float,title_train[cluster]))
		#dist = float(spatial.distance.euclidean(docs,temp))
		dist=cosin(docs,temp,covr)
		dist_clus.append(dist)
	clus_no = dist_clus.index(min(dist_clus))	
	Cclus_belong_view3_out.append(clus_no)

covr=[]
covr=np.array(venue_train)
covr=covr.T
covr=np.cov(covr)
for docs in venue_test:
	dist_clus = []
	for cluster in conscluscentre_view4:
		docs=list(map(float,docs))
		temp=[]
		temp=list(map(float,venue_train[cluster]))
		#dist = float(spatial.distance.euclidean(docs,temp))
		dist=cosin(docs,temp,covr)
		dist_clus.append(dist)
	clus_no = dist_clus.index(min(dist_clus))	
	Cclus_belong_view4.append(clus_no)

covr=[]
covr=np.array(venue_train)
covr=covr.T
covr=np.cov(covr)
for docs in venue_test_out:
	dist_clus = []
	for cluster in conscluscentre_view4:
		docs=list(map(float,docs))
		temp=[]
		temp=list(map(float,venue_train[cluster]))
		#dist = float(spatial.distance.euclidean(docs,temp))
		dist=cosin(docs,temp,covr)
		dist_clus.append(dist)
	clus_no = dist_clus.index(min(dist_clus))	
	Cclus_belong_view4_out.append(clus_no)

#-----------------------------------------------------------semantic-----------------------------------------------------
with open("resultdebo1.txt","a") as fout:
	fout.write("con semantic inscope\n")
covr=[]
covr=np.array(semantic_train)
covr=covr.T
covr=np.cov(covr)
for i,docs in enumerate(semantic_test):
	docs=list(map(float,docs))
	temp=[]
	temp=list(map(float,semantic_train[conscluscentre_view1[Cclus_belong_view1[i]]]))
	#temp=str(semantic_train[conscluscentre_view1[Cclus_belong_view1[i]]])
	#dist = model.wmdistance(docs,temp)
	dist=cosin(docs,temp,covr)
	if(dist!=1.0 and dist<(consdist_from_cent1[Cclus_belong_view1[i]])):
		Cinscope_view1.append(1)
	else:
		Cinscope_view1.append(0)
	with open("resultdebo1.txt","a") as fout:
		fout.write(str(dist)+"\n")
with open("resultdebo1.txt","a") as fout:
	fout.write("con semantic inscope end\n")
#-avg_consdist_from_cent1[Cclus_belong_view1[i]]+
with open("resultdebo1.txt","a") as fout:
	fout.write("con semantic outscope\n")
covr=[]
covr=np.array(semantic_train)
covr=covr.T
covr=np.cov(covr)
for i,docs in enumerate(semantic_test_out):
	docs=list(map(float,docs))
	temp=[]
	temp=list(map(float,semantic_train[conscluscentre_view1[Cclus_belong_view1_out[i]]]))
	#temp=str(semantic_train[conscluscentre_view1[Cclus_belong_view1_out[i]]])
	#dist = model.wmdistance(docs,temp)
	dist=cosin(docs,temp,covr)
	if(dist!=1.0 and dist<(consdist_from_cent1[Cclus_belong_view1_out[i]])):
		Coutscope_view1.append(1)
	else:
		Coutscope_view1.append(0)
	with open("resultdebo1.txt","a") as fout:
		fout.write(str(dist)+"\n")
with open("resultdebo1.txt","a") as fout:
	fout.write("con semantic outscope end\n")
#2*avg_consdist_from_cent1[Cclus_belong_view1_out[i]]+


#------------------------------------------------------------tfidf-------------------------------------------------------
with open("resultdebo1.txt","a") as fout:
	fout.write("con tfidf inscope\n")
covr=[]
covr=np.array(tfidf_train)
covr=covr.T
covr=np.cov(covr)
for i,docs in enumerate(tfidf_test):
	docs=list(map(float,docs))
	temp=[]
	temp=list(map(float,tfidf_train[conscluscentre_view2[Cclus_belong_view2[i]]]))
	#dist = float(spatial.distance.euclidean(docs,temp))
	dist=cosin(docs,temp,covr)
	if(dist!=1.0 and dist<(consdist_from_cent2[Cclus_belong_view2[i]])):
		Cinscope_view2.append(1)
	else:
		Cinscope_view2.append(0)
	with open("resultdebo1.txt","a") as fout:
		fout.write(str(dist)+"\n")
with open("resultdebo1.txt","a") as fout:
	fout.write("con tfidf inscope end\n")
#-avg_consdist_from_cent2[Cclus_belong_view2[i]]+
with open("resultdebo1.txt","a") as fout:
	fout.write("con tfidf outscope\n")
covr=[]
covr=np.array(tfidf_train)
covr=covr.T
covr=np.cov(covr)
for i,docs in enumerate(tfidf_test_out):
	docs=list(map(float,docs))
	temp=[]
	temp=list(map(float,tfidf_train[conscluscentre_view2[Cclus_belong_view2_out[i]]]))
	#dist = float(spatial.distance.euclidean(docs,temp))
	dist=cosin(docs,temp,covr)
	if(dist!=1.0 and dist<(consdist_from_cent2[Cclus_belong_view2_out[i]])):
		Coutscope_view2.append(1)
	else:
		Coutscope_view2.append(0)
	with open("resultdebo1.txt","a") as fout:
		fout.write(str(dist)+"\n")
with open("resultdebo1.txt","a") as fout:
	fout.write("con tfidf outscope end\n")

#2*avg_consdist_from_cent2[Cclus_belong_view2_out[i]]+

#------------------------------------------------------------title------------------------------------------------------
with open("resultdebo1.txt","a") as fout:
	fout.write("con title inscope\n")
covr=[]
covr=np.array(title_train)
covr=covr.T
covr=np.cov(covr)
for i,docs in enumerate(title_test):
	docs=list(map(float,docs))
	temp=[]
	temp=list(map(float,title_train[conscluscentre_view3[Cclus_belong_view3[i]]]))
	#dist = float(spatial.distance.euclidean(docs,temp))
	dist=cosin(docs,temp,covr)
	if(dist!=1.0 and dist<(consdist_from_cent3[Cclus_belong_view3[i]])):
		Cinscope_view3.append(1)
	else:
		Cinscope_view3.append(0)
	with open("resultdebo1.txt","a") as fout:
		print(str(dist)+"\n")
with open("resultdebo1.txt","a") as fout:
	fout.write("con title inscope end\n")
#-avg_consdist_from_cent3[Cclus_belong_view3[i]]+
with open("resultdebo1.txt","a") as fout:
	fout.write("con title outscope\n")
covr=[]
covr=np.array(title_train)
covr=covr.T
covr=np.cov(covr)
for i,docs in enumerate(title_test_out):
	docs=list(map(float,docs))
	temp=[]
	temp=list(map(float,title_train[conscluscentre_view3[Cclus_belong_view3_out[i]]]))
	#dist = float(spatial.distance.euclidean(docs,temp))
	dist=cosin(docs,temp,covr)
	if(dist!=1.0 and dist<(consdist_from_cent3[Cclus_belong_view3_out[i]])):
		Coutscope_view3.append(1)
	else:
		Coutscope_view3.append(0)
	with open("resultdebo1.txt","a") as fout:
		fout.write(str(dist)+"\n")
with open("resultdebo1.txt","a") as fout:
	fout.write("con title outscope end\n")
#2*avg_consdist_from_cent3[Cclus_belong_view3_out[i]]+

#------------------------------------------------------------venue---------------------------------------------------------
with open("resultdebo1.txt","a") as fout:
	fout.write("con venue inscope\n")
covr=[]
covr=np.array(venue_train)
covr=covr.T
covr=np.cov(covr)
for i,docs in enumerate(venue_test):
	docs=list(map(float,docs))
	temp=[]
	temp=list(map(float,venue_train[conscluscentre_view4[Cclus_belong_view4[i]]]))
	#dist = float(spatial.distance.euclidean(docs,temp))
	dist=cosin(docs,temp,covr)
	if(dist!=1.0 and dist<(consdist_from_cent4[Cclus_belong_view4[i]])):
		Cinscope_view4.append(1)
	else:
		Cinscope_view4.append(0)
	with open("resultdebo1.txt","a") as fout:
		fout.write(str(dist)+"\n")
with open("resultdebo1.txt","a") as fout:
	fout.write("con venue inscope end\n")
#-avg_consdist_from_cent4[Cclus_belong_view4[i]]+
with open("resultdebo1.txt","a") as fout:
	fout.write("con venue outscope\n")
covr=[]
covr=np.array(venue_train)
covr=covr.T
covr=np.cov(covr)
for i,docs in enumerate(venue_test_out):
	docs=list(map(float,docs))
	temp=[]
	temp=list(map(float,venue_train[conscluscentre_view4[Cclus_belong_view4_out[i]]]))
	#dist = float(spatial.distance.euclidean(docs,temp))
	dist=cosin(docs,temp,covr)
	if(dist!=1.0 and dist<(consdist_from_cent4[Cclus_belong_view4_out[i]])):
		Coutscope_view4.append(1)
	else:
		Coutscope_view4.append(0)
	with open("resultdebo1.txt","a") as fout:
		fout.write(str(dist)+"\n")
with open("resultdebo1.txt","a") as fout:
	fout.write("con venue outscope end\n")
#2*avg_consdist_from_cent4[Cclus_belong_view4_out[i]]+

print("cons")


print(Cinscope_view1)
print(Cinscope_view2)
print(Cinscope_view3)
print(Cinscope_view4)
print(Coutscope_view1)
print(Coutscope_view2)
print(Coutscope_view3)
print(Coutscope_view4)





		






closest_doc_list = []

#for cl_view1,cl_view2,cl_view3,cl_view4 in zip(clus_belong_view1,clus_belong_view2,clus_belong_view3,clus_belong_view4):
#	doc_list = mem1_clus[cl_view1] + mem2_clus[cl_view2] +mem3_clus[cl_view3] + mem4_clus[cl_view4]
#	closest_doc_list.append(doc_list)

#most_common_point = []
#print("closest doc kist")
#print(closest_doc_list)
#for doc_close_point in closest_doc_list:
#	most_common,num_most_common = Counter(doc_close_point).most_common(1)
	#print("counter")
	#print(Counter(doc_close_point).most_common(1))
	#print("----------")
	#print(most_common)
	#print("----------")
#	most_common_point.append(most_common)
#print(most_common_point)
#doc_final_belong = []
#for docs in most_common_point:
#	for i,clusters in enumerate(mem_consensus):
#		if(clusters[docs]==1.00):
#			doc_final_belong.append(i)
#print(doc_final_belong)
#with open("Cluster_belonging.txt","w") as results:
#	for lines in doc_final_belong:
#		results.write(str(lines)+"\n")
#		results.write("\n") 			
