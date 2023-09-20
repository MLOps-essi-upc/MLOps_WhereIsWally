# MLOps_WhereIsWally

# Dataset Card for WhereIsWally Object Detection 

## Dataset Description

- **Homepage:** 
- **Repository:** 
- **Paper:** 
- **Leaderboard:** 
- **Point of Contact:** 

### Dataset Summary

The WhereIsWally dataset contains 3514 snapshots of different scenarios of the 17 existing Where Is Wally books. Each scenario contains one or more of the following characters: Odlaw, Wally, Wilma and Wizard. Annotations of where none, one or more of these characters can be found are added to each snapshot which facilitates building an objecr detection model.

### Supported Tasks and Leaderboards

This dataset support building an object detection model to find the character(s). 

### Languages

As it is a graphical dataset, language presence does not apply.

## Dataset Structure

### Data Instances

A typical image is divided up in different snapshots. Each of the snapshots contains none, one or more annoted characters. 

### Data Fields

[More Information Needed]

### Data Splits

|    Train    | Validation |    Test    |
| ----------- | ---------- | ---------- |
| 2811 (80%)  | 351 (10%)  | 352 (10%)  |

## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

This dataset is a Roboflow project called "Wally v3 Computer Vision Project". You can find references below:
Reference: Wally, “Wally v3 dataset.” https://universe.roboflow.com/wally/wally-v3 ,
mar 2022. visited on 2023-09-20.

```
BibTex: @misc{ wally-v3_dataset,
    title = { Wally v3 Dataset },
    type = { Open Source Dataset },
    author = { Wally },
    howpublished = { \url{ https://universe.roboflow.com/wally/wally-v3 } },
    url = { https://universe.roboflow.com/wally/wally-v3 },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2022 },
    month = { mar },
    note = { visited on 2023-09-20 },
}
```

link: https://universe.roboflow.com/wally/wally-v3

### Annotations

#### Annotation process

None, one or more annotations per snapshot

#### Who are the annotators?

Roboflow user: Wally (https://universe.roboflow.com/wally)

## Considerations for Using the Data

### Social Impact of Dataset

Recreational Purposes

## Additional Information

### Dataset Curators

Roboflow user: Wally (https://universe.roboflow.com/wally)
