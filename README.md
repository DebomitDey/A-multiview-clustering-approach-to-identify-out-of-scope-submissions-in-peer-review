# A-multiview-clustering-approach-to-identify-out-of-scope-submissions-in-peer-review
There are three parts for the whole scope detection project ,they are as follows :

# 1. Views Generation
FILE NAME: b2MVMOO_2.py
1) Views generation : We generate four views , 1 lexical(TF-IDF matrix),2  Bibliographic(title and venue) and 1 semantic .On the basis of 4 views 4 corresponding distance matrices are generated.
#One need to change the "base2" value to where the trainig data is there.
#"base3" value needs to be set to the directory where test "inscope" data is there.
#"base4" value needs to be set to the directory where test "outscope" data is there.

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

#compile the code as : g++ amosa_mores_copy.cpp -o amosa_mores_copy
#run the code as : ./amosa_mores_copy Dist_tfidf.txt Dist_venue.txt Dist_title.txt Dist_semantic resultFile (size of training datset)

Here size of training dataset refers to the no f documents used for training.(Check it from the distance matrix)  

Output : Gives four membership matrices (which are clustering matrices for each of the four views)  and one consensus partitioning :

#Membership_consensus.txt
#Membership1.txt
#Membership2.txt
#Membership3.txt
#Membership4.txt
# 3. Scope detection
FILE NAME : bScope.py

It checks whether the test instance is incope or out of scope and gives the precision recall scores.

It requires the following matrices as input :

#Tfidf_train.txt ,Tfidf_test.txt and Tfidf_test_out.txt
#Title_train.txt ,Title_test.txt and Title_test_out.txt
#Venue_train.txt ,Venue_test.txt and Venue_test_out.txt
#Semantic_train.txt ,Semantic_test.txt and and Semantic_test_out.txt
#Dist_venue.txt
#Dist_tfidf.txt
#Dist_title.txt
#Dist_semantic.txt
#Membership_consensus.txt
#Membership1.txt , Membership2.txt ,Membership3.txt ,Membership4.txt  

Output :  Gives the list of the documents belonging to the respective cluster no of the journal (0 in case if the document doesn't belong to any cluster)




# About the algorithm
The real-coded Archived Multi Objective Simulated Annealing (AMOSA) is introduced and develop by the writers of the following paper:

    Authors: Sanghamitra Bandyopadhyay, Sriparna Saha, Ujjwal Maulik and Kalyanmoy Deb.
    Paper Title: A Simulated Annealing Based Multi-objective Optimization Algorithm: AMOSA
    Journal: IEEE Transaction on Evolutionary Computation, Volume 12, No. 3, JUNE 2008, Pages 269-283.

The original code is available for the download from Sriparna Saha's website at: http://www.isical.ac.in/~sriparna_r

# How to Compile and Run
