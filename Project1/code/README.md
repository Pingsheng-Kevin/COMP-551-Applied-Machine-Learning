# How to run the scripts
First two csv files must set in the same directory with these scripts

For experiments related to KNN, use:

python main.py

For experiments related to Decision tree, use:

python Decision_Tree_experiments.py -c breast_cancer_wisconsin.csv

or

python Decision_Tree_experiments.py -H hepatitis.csv

to run experiments on different datasets.

to show results of preprocessing, run:

python Data_preprocessing.py -c/-H breast_cancer_wisconsin.csv/hepatitis.csv

to generate decision boundaries related to decision tree, run:

python Decision_Tree_plot_generator -c/-H breast_cancer_wisconsin.csv/hepatitis.csv

Warnings: The command above will generate every possible combination between features, so it takes very long time.

to generate plots related to experiments in the report, run:

python analysis_plot_generator.py

