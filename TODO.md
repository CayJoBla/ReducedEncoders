

## Organization
* Fix the disorganization in both git repos
* Organize the output for the SBERT reduced model
* Define reduced model output classes (?)
* Group models of similar types in a single file
* Rename the package to something more general (e.g. `reduced_models` or `transformers_reduced`)
* Add an AutoReducedModel class that can be used to automatically load the correct reduced model class based on the model type
* Separate the package into a separate git repo (?)

## SBERT Reduced
* Write a training script for SBERT reduced
* Evaluate SBERT on the GLUE tasks

## BERT Reduced
* Make the local BERT reduced model consistent with hub model
* Fix the problems with the GLUE submission