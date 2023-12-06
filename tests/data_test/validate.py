"""Module for great expectations """
import os

import great_expectations as gx

context = gx.get_context()


def validate_data(validator, images_list):
    """Function validate_data"""
    validator.expect_column_values_to_not_be_null("filename")
    validator.expect_column_values_to_not_be_null("width")
    validator.expect_column_values_to_not_be_null("height")
    validator.expect_column_values_to_not_be_null("class")
    validator.expect_column_values_to_not_be_null("xmin")
    validator.expect_column_values_to_not_be_null("ymin")
    validator.expect_column_values_to_not_be_null("xmax")
    validator.expect_column_values_to_not_be_null("ymax")

    validator.expect_column_values_to_be_of_type("filename", "object")
    validator.expect_column_values_to_be_of_type("width", "int")
    validator.expect_column_values_to_be_of_type("height", "int")
    validator.expect_column_values_to_be_of_type("class", "object")
    validator.expect_column_values_to_be_of_type("xmin", "int")
    validator.expect_column_values_to_be_of_type("ymin", "int")
    validator.expect_column_values_to_be_of_type("xmax", "int")
    validator.expect_column_values_to_be_of_type("ymax", "int")

    validator.expect_column_values_to_be_between("width", min_value=0, max_value=1280)
    validator.expect_column_values_to_be_between("height", min_value=0, max_value=1280)
    validator.expect_column_values_to_be_between("xmin", min_value=0, max_value=1280)
    validator.expect_column_values_to_be_between("ymin", min_value=0, max_value=1280)
    validator.expect_column_values_to_be_between("xmax", min_value=0, max_value=1280)
    validator.expect_column_values_to_be_between("ymax", min_value=0, max_value=1280)

    validator.expect_column_values_to_be_unique("filename")

    validator.expect_column_distinct_values_to_be_in_set("class",
                                                         {"Odlaw", "Wizard", "Wilma", "Wally"})

    # checking annotation filenames to exists in data dir
    validator.expect_column_distinct_values_to_be_in_set("filename",
                                                         images_list)

    # checking filenames in dir to exist in annotation
    validator.expect_column_distinct_values_to_contain_set("filename",
                                                           images_list)

    validator.save_expectation_suite()

    checkpoint = context.add_or_update_checkpoint(
        name="train_checkpoint",
        validator=validator,
    )
    checkpoint_result = checkpoint.run()
    context.view_validation_result(checkpoint_result)


# TRAIN DATA
DATA_PATH = "../../data/raw/train/_annotations.csv"
train_validator = context.sources.pandas_default.read_csv(DATA_PATH)
images = os.listdir("../../data/raw/train/")
validate_data(train_validator, images)

# VALIDATION DATA
DATA_PATH = "../../data/raw/valid/_annotations.csv"
train_validator = context.sources.pandas_default.read_csv(DATA_PATH)
images = os.listdir("../../data/raw/valid/")
validate_data(train_validator, images)

# TEST DATA
DATA_PATH = "../../data/raw/test/_annotations.csv"
train_validator = context.sources.pandas_default.read_csv(DATA_PATH)
images = os.listdir("../../data/raw/test/")
validate_data(train_validator, images)
