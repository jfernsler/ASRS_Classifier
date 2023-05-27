## Aviation Safety Reporting System (ASRS) Classifier
This is a language model fine tuned to classify ASRS Narrative data into multiple anomaly catagories. Anomalies in ASRS reports can contain any number of various options, so this project is an effort to reduce those to managable collections of common issues. 

### ASRS Reports
An ASRS report is filed by pilots or staff of an aircraft when an anomaly is encountered. The anomaly is logged as a collection of options and sub-selections, additional data about the flight and crew are collected, and finally a narrative is written by the submitter. 

A sample Anomaly / Narrative looks like this:

    Anomaly: 
    deviation - altitude excursion from assigned altitude. deviation, discrepancy - procedural clearance. 
    deviation, discrepancy - procedural other, unknown. 
    other similar soonding call signs
    
    Narrative: 
    we started to dsnd from 15000 ft to 11000 ft. about 13500 ft, apch asked our alt. 
    we responded 13500 ft to 11000 ft. he said we were supposed to be at 15000 ft. 
    he clred us to 13000 ft. no conflicts, as far as we know. in the next few mins, 
    we discovered we read a clrnc for someone else in our company. the misunderstanding 
    resolved in our number being abc while company bc was on same freq. company bc thought 
    the call was for us too, and had remained at their alt. it would help if flt numbers 
    did not end in the same number.

### The Dataset
This project uses an ASRS dataset from kaggle found here:

https://www.kaggle.com/datasets/andrewda/aviation-safety-reporting-system


## Data Cleanup
For training I used only data from the year 2000 and on - which contains over 125,000 entries. Text was cast to lower case and some punctuation was converted and removed. Furthermore I am only using the Narratives and Anomalies in the project. 

Included in this repository are only the test datasets for size considerations.

### Clustering
The reduced data contains nearly 40,000 unique anomaly listings. These were all converted to text embeddings (using a BERT model) and clustered with KMeans by similarity into 15 groups. These groups (named 0-14) were used as labels for the narrative data.

### Classification
Finally the narrative data was used to fine tune a BERT classifer on the 15 labels.

### Results
Given limited hardware and training time, the model achieves roughly 50% success rate in classifications. Random choice across 15 labels would yield 6.6% success. 



