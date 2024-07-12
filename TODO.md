# Repo
## Organization
* Switch SequenceClassifierOutput to ReducedSequenceClassifierOutput (?)
* Tidy up reduced output classes
* Separate the package into a separate git repo (?)

# Models
## SBERT Reduced
* Try submitting the SBERT reduced model to GLUE
## BERT Reduced
* The "glue" revision of the reduced BERT model is broken. May need to retrain ReducedBERT on GLUE tasks.
* Fix the problems with the GLUE submission
## CompreSBERT
* Test the model on the GLUE tasks
* Test the model on clustering / topic modeling
* Try clustering on other data
* Submit the model for GLUE tasks
## Compressed SBERT PUMAP model
* Finish embedding the wikipedia dataset
* Train the reduction / expansion layers with PUMAP
* Test the model on language tasks
* Test the model on clustering / topic modeling

# Clustering
## DenStream
* Look at modifying the DenStream algorithm to make more sense for our data
* Think of what needs to be adapted so that the online micro-clusters work with the I-HASTREAM algorithm
## I-HASTREAM
* Start implementing the I-HASTREAM algorithm, keeping our use case in mind
* Make sure that it will work with the DenStream online micro-clusters if we want to go that route

# Topic Modeling
## c-TF-IDF
* Incorporate the c-TF-IDF algorithm into the clustering algorithm so that it can be done online

# Evaluation
## Metrics
* Look at potential metrics for streamed clustering and topic modeling
* ex. Temporal Silhouette, ... (?)
## Synthetic / Test Data
* See how the encoding, clustering, and topic modeling algorithms work on synthetic data individually
* Test together as a whole and get NUMERICAL results
## Real Data
* Start to implement the full algorithm on Twitter(X) data
* Look into parameter tuning for the algorithms

# Administrative
## Paper
* Look at starting the paper (reduction, topic modeling)
* Look for where to publish the paper (not super big, not AAAI)
