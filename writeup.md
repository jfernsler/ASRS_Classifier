___

## Pitch: 
Aviation Safety Reporting System (ASRS) logs are filed for every incident that could occur during a flight and can be filed by any member of the crew - these human generated reports contain a vast number of optional combinations of anomalies as well as a written narrative describing the event. The dense reports could be narrowed down to fewer categories to allow them to be addressed in a more meaningful order and place resources on issues which require more attention earlier. I propose a classification system based on the narrative and clustered by anomaly types with the aim to decouple the reporting methods - this proof of concept show the potential of this process and could further benefit from more input from professionals to craft and prioritize anomaly sets for more clarity.

## Data source:
This project uses an ASRS dataset from kaggle found here: https://www.kaggle.com/datasets/andrewda/aviation-safety-reporting-system. This 580+Mb csv file contains over 250,000 reports which was reduced to 125,000 by using only reports from the year 2000-2022. Code was written to create text embeddings from the 40,000+ unique anomaly reports and clusted into 15 classes (although 15 is a somewhat arbitrary value, it yielded better clustering than some other options and additional future work would allow for professional guidedence to help refine and name these clusters), all class numbers were then paired with the narratives for training and evaluation.

## Model and data justification:
A smaller version of a BERT model was chosen (distilBERT) in order to leverage its excellent contextual understanding of text and its strength in multi-label classification. It was used for both the anomaly embedding generation as well as being fine-tuned for the classification task where the narrative was used as input. The final tuned model was uploaded to the huggingface.co hub here:  https://huggingface.co/jfernsler/ASRS_distilbert-base-uncased

## Commented examples:
example 1 - Inference at narrative index 123 (```a3_main.py -po 123```):
```
Narrative at index 123:
 on may/mon/04, given job case 32-96-50 to replace #2 tire. 
 removed and replaced tire, torqued and spun tire. installed 
 lock screws and safetied.

Anomaly:
 aircraft equipment problem critical. deviation, discrepancy - 
 procedural far. deviation, discrepancy - procedural maintenance. 
 deviation, discrepancy - procedural published material, policy.

True Label: 7
Predicted Label: 7 (Confidence: 0.93)
```

Here the narrative is being correctly assesed into the anomaly cluster '7'. Here is a sampling of anomalies from this cluster:

* aircraft equipment problem critical. deviation, discrepancy - procedural published material, policy. flight deck, cabin, aircraft event smoke, fire, fumes, odor.
* aircraft equipment problem less severe. deviation, discrepancy - procedural published material, policy. deviation, discrepancy - procedural mel, cdl. deviation, discrepancy - procedural far. inflight event, encounter fuel issue.
* aircraft equipment problem critical. deviation, discrepancy - procedural far. deviation, discrepancy - procedural maintenance. deviation, discrepancy - procedural published material, policy.
* aircraft equipment problem less severe. deviation - altitude crossing restriction not met. deviation - altitude undershoot. deviation, discrepancy - procedural published material, policy. deviation, discrepancy - procedural mel, cdl. deviation, discrepancy - procedural far. deviation, discrepancy - procedural clearance. inflight event, encounter weather, turbulence. other exp lvl tech flying

_CORRECT PREDICTION_. The model is correctly assess the narrative as an equipment problem. Additionally one can see how the combination of anomaly selections can create a huge potential of issues to wade through. Given he 'critical' and 'less severe' tags in the anomaly cluster, this illustrates how professional finetuning and labeling would help this system.
___
example 2 - Inference at narrative index 55 (```a3_main.py -po 55```):
```
Narrative at index 55:
 first officer was on base leg to btv and descending to traffic 
 pattern altitude (tpa) of 1;800 ft msl when passing through 2;000 ft 
 msl, the egpws 'terrain' aural alert was issued. the first officer 
 stopped the descent and initiated a shallow climb to clear the aural 
 alert, leveled momentarily, then resumed the scheduled descent to 
 the field with no further incident. both the captain and first 
 officer verified that no terrain on the mfd was in conflict with 
 the aircraft and a visual check of the surrounding terrain was 
 verified. no further alerts were issued and the approach was completed 
 per normal company profile.

Anomaly:
 aircraft equipment problem less severe.

True Label: 13
Predicted Label: 5 (Confidence: 0.72)
```

Anomaly samples from cluster '5' - the prediction:
* inflight event, encounter other, unknown.
* inflight event, encounter weather, turbulence. inflight event, encounter unstabilized approach. inflight event, encounter cftt, cfit.
* ground event, encounter other, unknown.

_INCORRECT PREDICTION_. The model has predicted that the issue occured in flight - which is a correct assumption. The mis-categorization could be a result of the somewhat cross-correlated anomaly sets. This illustrates the possibility of anomaly selections that are 'different but the same' and highlights the potential of a method for distilling categories from the narratives themselves.

## Testing:
With 15 potential class choices, a purely random selection would yield a 6.6% success rate. This model, fine-tuned with only 3 epochs due to time and hardware constraints, has a roughly 54% accuracy, showing a strong diagonal in the confusion matrix. The training loss curve indicates that there is still room to train and improve the accuracy of them model. Also displayed here is a 3D visualization of the anomaly clusters.

## Code and instructions to run it:
The code and a reduced dataset can be cloned from: 
* https://github.com/jfernsler/ASRS_Classifier

Once cloned, the primary script to run is a3_main.py. Run this script with one of the follwing flags:
* ```a3_main.py -po [idx]```
    * --predict_one
    * Evaluates a single narrative. If no index is given one will be randomly selected from the test dataset. The test dataset contains around 25,000 records.
* ```a3_main.py -pm [count]```
    * --predict_many
    * Evaluates a collection of randomly selected narratives from the test dataset. If no count is given, 10 will be selected. This method also counts correct results as well as making 'random' choices along the way with tallies shown at the end.
* ```a3_main.py -d [idx]```
    * --test_dataset
    * Extracts a single data point and displays the information associated. If no index is given a data row will be selected at random from the test dataset.

Please note, the trained model has been placed into the huggingface.co hub (https://huggingface.co/jfernsler/ASRS_distilbert-base-uncased) and *should* automatically download when the inference is first run. The model size is roughly 280Mb. 

## Additional Information
* ```data_out/cluster_samples.txt``` contains anomaly samples from each cluster.
* To run training several steps must be taken:
    * the original asrs data must be downloaded and placed in the data folder as ```asrs.csv```
    * ```asrs_create_clusters.py``` needs to be run which will embed and cluster the anomalies as well as generate the test and train datasets.
    * ```asrs_preprocess.py``` needs to be run, which will generate embeddings for the narratives across the train and test datasets and save those out to pickle files.
    * ```asrs_train_multi_classifier.py``` can then be run. Be mindful that the distilBERT model is fairly large and requires a decently scoped GPU to run. This was was trained on a 6Gb nvidia 1060Ti and could only manage a batch size of 8. 
