# A-multiview-clustering-approach-to-identify-out-of-scope-submissions-in-peer-review
There are three parts for the whole scope detection project ,they are as follows :

# 1. Views Generation
FILE NAME: b2MVMOO_2.py
#run the code as : python3 b2MVMOO_2.py
1) Views generation : We generate four views , 1 lexical(TF-IDF matrix),2  Bibliographic(title and venue) and 1 semantic .On the basis of 4 views 4 corresponding distance matrices are generated.
#One need to change the "base2" value to where the trainig data is there.
#"base3" value needs to be set to the directory where test "inscope" data is present.
#"base4" value needs to be set to the directory where test "outscope" data is present.

We get the corresponding views names as:

	1) Dist_venue.txt (Venue distance matrix) 
	2) Dist_title.txt (Title distance matrix)	
	3) Dist_tfidf.txt (Lexical distance matrix)	
	4) Dist_semantic.txt (Semantic distance matrix)

Along with the distance matrices this code also stores feature matrices for future references.They are :

#TFIDF matrices : Tfidf_train.txt and Tfidf_test.txt
#Title matrices : Title_train.txt and Title_test.txt
#Venue matrices : Venue_train.txt and Venue_test.txt
#Semantic matrices : Semantic_train.txt and Semantic_test.txt
# 2. Multiview Clustering
FILE NAME : amosa_mores_copy.cpp 
Here the clustering part is done on the distance matrices.
Inside training dataset there may be some clusters. A data is said to be out of scope if it does not lie in vicinity of any of the clusters formed by the training data.

#compile the code as : g++ amosa_mores_copy.cpp -o amosa_mores_copy
#run the code as : ./amosa_mores_copy Dist_tfidf.txt Dist_venue.txt Dist_title.txt Dist_semantic resultFile (size of training datset)

Here size of training dataset refers to the no of documents used for training.(Check it from the distance matrix)  

Output : Gives four membership matrices (which are clustering matrices for each view)  and  consensus partitionings (there may be several consensus partitionings) :

#resultFile_journal_clusi_multiview (here i=1 to i=n . Here n will vary depending on how many consensus partitionings are created)
#Membership1.txt
#Membership2.txt
#Membership3.txt
#Membership4.txt



Here we will take the consensus partitions of the train data. And here we shall create a membership matrix depicting a data belongs to which cluster corresponding to each consensus partition.

FILE NAME : consensus_membership.py
#run the code as : python3 consensus_membership.py
Input : resultFile_journal_clusi_multiview (here i=0 to i=n)
Output : membership_consensusi.txt (here i=0 to i=n)


# 3. Scope detection
FILE NAME : Bscope.py
#run the code as : python3 Bscope.py

For each test data it calculates the minimum distance from a partition in terms of tfidf distance,semantic distance and venue distance and we will use these three distances as the new feature of the test data.  

It requires the following matrices as input :

#Tfidf_train.txt ,Tfidf_test.txt and Tfidf_test_out.txt
#Title_train.txt ,Title_test.txt and Title_test_out.txt
#Venue_train.txt ,Venue_test.txt and Venue_test_out.txt
#Semantic_train.txt ,Semantic_test.txt and and Semantic_test_out.txt
#Dist_venue.txt
#Dist_tfidf.txt
#Dist_title.txt
#Dist_semantic.txt
#membership_consensus1.txt (here we take only one consensus partition of the training data to determine the scope of the test data)

#Output : resultdebo1.txt



Then with the help of new features we apply k-medoid algorithm to classify the inscope and outscope data.

FILE NAME : resplot.py
#run the code as : python3 resplot.py
here inside the resplot.py put the consensus semantic distances, consensus tfidf distances, consensus venue distances from resultdebo1.txt in  a,b,e list(for inscope data) and a1,b1.e1 list(for outscope data) respectively
Output :  Gives 0 or 1 for each test data where 0 means inscope and 1 means outscpe data and the accuracy,precision,recall and f1 scores.




# About the algorithm
The real-coded Archived Multi Objective Simulated Annealing (AMOSA) is introduced and develop by the writers of the following paper:

    Authors: Sanghamitra Bandyopadhyay, Sriparna Saha, Ujjwal Maulik and Kalyanmoy Deb.
    Paper Title: A Simulated Annealing Based Multi-objective Optimization Algorithm: AMOSA
    Journal: IEEE Transaction on Evolutionary Computation, Volume 12, No. 3, JUNE 2008, Pages 269-283.



