## Organization
* Switch SequenceClassifierOutput to ReducedSequenceClassifierOutput (?)
* Tidy up reduced output classes
* Separate the package into a separate git repo (?)

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

## Clustering algorithm
* Explore ways to speed up online clustering with HDBSCAN
* Look into other clustering algorithms (can they be done online?); DBSCAN, OPTICS, etc.

## Administrative
* Look at starting the paper (reduction, topic modeling)
* Look for where to publish the paper (not super big, not AAAI)
