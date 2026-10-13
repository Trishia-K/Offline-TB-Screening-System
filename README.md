# Offline TB Screening System

Final year project — BSc Computer Science
Uganda Christian University  
Kobumanzi Trishia (M24B23/011)

## What this is

A simple offline tool to help screen for tuberculosis using a chest X-ray and a few symptom questions. It is meant for health centres that do not always have a radiologist or stable internet.

The system does not diagnose TB as a final tool but flags patients who should be referred for sputum testing.

## What it does now

- Loads a MobileNetV2 model trained on the public Kaggle TB chest X-ray dataset
- Takes an uploaded chest X-ray and six yes/no symptom answers
- Gives a combined score and a Class Activation Mapping heatmap
- Saves patient and screening records locally (SQLite)
- Runs fully offline in a Flask web app (no cloud needed for prediction)

## What was tested

- Compared MobileNetV2 and VGG16 on the same data split (MobileNetV2 kept because of higher sensitivity)
- Trained a U-Net for lung segmentation; using segmented images did not improve classification, so segmentation is not used in the final pipeline
- Symptom score uses fixed weights based on common TB screening criteria

## Planned next steps

- External check on the Shenzhen dataset
- Measure inference time and model size
- TensorFlow Lite conversion if useful
- Clearer printable referral report

