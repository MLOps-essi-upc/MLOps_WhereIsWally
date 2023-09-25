




# Model Card for 

The model performs multi-class object detection usign the algorithm YOLO v7. 


#  Table of Contents

- [Model Card for ](#model-card-for--model_id-)
- [Table of Contents](#table-of-contents)
- [Table of Contents](#table-of-contents-1)
- [Model Details](#model-details)
  - [Model Description](#model-description)
- [Uses](#uses)
  - [Direct Use](#direct-use)
  - [Downstream Use [Optional]](#downstream-use-optional)
  - [Out-of-Scope Use](#out-of-scope-use)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
  - [Recommendations](#recommendations)
- [Training Details](#training-details)
  - [Training Data](#training-data)
  - [Training Procedure](#training-procedure)
    - [Preprocessing](#preprocessing)
    - [Speeds, Sizes, Times](#speeds-sizes-times)
- [Evaluation](#evaluation)
  - [Testing Data, Factors & Metrics](#testing-data-factors--metrics)
    - [Testing Data](#testing-data)
    - [Factors](#factors)
    - [Metrics](#metrics)
  - [Results](#results)
- [Model Examination](#model-examination)
- [Environmental Impact](#environmental-impact)
- [Technical Specifications [optional]](#technical-specifications-optional)
  - [Model Architecture and Objective](#model-architecture-and-objective)
  - [Compute Infrastructure](#compute-infrastructure)
    - [Hardware](#hardware)
    - [Software](#software)
- [Citation](#citation)
- [Glossary [optional]](#glossary-optional)
- [More Information [optional]](#more-information-optional)
- [Model Card Authors [optional]](#model-card-authors-optional)
- [Model Card Contact](#model-card-contact)
- [How to Get Started with the Model](#how-to-get-started-with-the-model)


# Model Details

## Model Description

The goal of this project is to solve Where is Wally puzzles by finding the exact position of Wally and the other characters (Wenda, Odlaw, Wizard and Woof).  
We are going to use a model pretrained on MS-COCO for object detection followed by transfer learning.

YOLO (You Only Look Once) is a popular object detection model known for its speed and accuracy. It introduces an end-to-end neural network that simultaneously predicts bounding boxes and class probabilities. This contrasts with earlier object detection methods that adapted classifiers for detection tasks. Adopting this novel approach, YOLO surpassed the performance of other real-time object detection systems by a significant measure.

- **Developed by:** Fatima Zohra Chiriki, Ange Xu, Ximena Moure, Louis Van Langendonck, Sebastian Paglia
- **Model type:** Supervised learning model
- **Language(s):** English
- **License:** More information needed
- **Parent Model:** Finetuned from Model: TODO
- **Resources for more information:** More information needed



# Uses

The model designed to solve "Where is Wally?" puzzles would be mainly used for the following purposes:
- Entertaining: the model can be used to create entertaining applications to perform the following tasks: finding wally and other hidden characters in complex scenes, verifying automatically if the players has found or not the hidden characters and also for generating new puzzles.
  
- Education: in educational contexts the model can be used to develop interactive learning materials for children. It can be adapted to teach kids various skills like visual perception, attention to deal and proble-solving in a fun and engaging way.

- Quality control: Publishers of puzzle books, magazines or newspapers could use the model as a quality control tool, as it can verify if the Wally appears in the scenes or not before their publication.
- 
## Direct Use
- The model can be used for task of Multi-class Object Detection.

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->
<!-- If the user enters content, print that. If not, but they enter a task in the list, use that. If neither, say "more info needed." -->


## Downstream Use [Optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->
<!-- If the user enters content, print that. If not, but they enter a task in the list, use that. If neither, say "more info needed." -->
 
Tasks that leverage this model include: Entertaining applications developement, Quality control evaluator, Real-time Object Tracking,etc.

## Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->
<!-- If the user enters content, print that. If not, but they enter a task in the list, use that. If neither, say "more info needed." -->

List of some out-of-scope uses:
- General Object Detection: The model is not designed for common object detection tasks, such as identifying people or animals in the image.

- Real-time Object Tracking: It not designed for real-time object tracking in videos, only for static images.


# Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->
The model may not be able to locate the hidden characters. Like in all AI models, false positives and false negatives could occur with a lower probability.


## Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

The users should be aware of the bias, risk, and technical limitations. Allow users to report inaccuracies and use their feedback for model improvement.


# Training Details

## Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

More information on training data needed


## Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

### Preprocessing

More information needed

### Speeds, Sizes, Times

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

More information needed
 
# Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

## Testing Data, Factors & Metrics

### Testing Data

<!-- This should link to a Data Card if possible. -->

More information needed


### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

More information needed

### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

More information needed

## Results 

More information needed

# Model Examination

More information needed

# Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** More information needed
- **Hours used:** More information needed
- **Cloud Provider:** More information needed
- **Compute Region:** More information needed
- **Carbon Emitted:** More information needed

# Technical Specifications [optional]

## Model Architecture and Objective

More information needed

## Compute Infrastructure

More information needed

### Hardware

More information needed

### Software

More information needed

# Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

More information needed

**APA:**

More information needed

# Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

More information needed

# More Information [optional]

More information needed

# Model Card Authors [optional]

<!-- This section provides another layer of transparency and accountability. Whose views is this model card representing? How many voices were included in its construction? Etc. -->

More information needed

# Model Card Contact

More information needed

# How to Get Started with the Model

Use the code below to get started with the model.

<details>
<summary> Click to expand </summary>

More information needed

</details>
