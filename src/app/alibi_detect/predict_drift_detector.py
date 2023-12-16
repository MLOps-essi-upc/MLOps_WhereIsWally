import numpy as np
from os import listdir
from os.path import isfile, join
from src import RAW_DATA_DIR,LOAD_DRIFT_DETECTOR_DIR
from alibi_detect.saving import load_detector
import datetime

model=load_detector(LOAD_DRIFT_DETECTOR_DIR)

def predict(img):
    image=np.asarray(img).astype('float32') / 255.
    image=np.expand_dims(image, 0)
    
    #inference
    model.infer_threshold(image, threshold_perc=95)
    preds = model.predict(image, outlier_type='instance',
                return_instance_score=True,
                return_feature_score=True)

    n_outliers=np.count_nonzero(preds['data']['is_outlier'] == 1)
    print("n outliers",n_outliers)

    # ct stores current time
    ct = datetime.datetime.now()
    # lgo the results
    f = open("log.txt", "a")
    f.write(str(ct)+"\t"+str(n_outliers))
    f.close()

