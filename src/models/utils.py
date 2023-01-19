import os


def count_files(path):
    """ Count files in directory and return number"""
    count = 0
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            count += 1
    return count


def get_latest_model(model_dir="models/trained_models"):
    """ Return path to latest model in directory, based on _number in filename"""
    print("Getting latest model")
    latest_model = None
    latest_number = 0
    for file in os.listdir(model_dir):
        if os.path.isfile(os.path.join(model_dir, file)):
            if file.startswith("model_checkpoint_"):
                number = int(file.split("_")[-1].split(".")[0])
                if number > latest_number:
                    latest_number = number
                    latest_model = os.path.join(model_dir, file)
    return latest_model
