import os
import pickle


def make_directory(directory):
    if not (os.path.exists(directory)):
        os.makedirs(directory)


def save_object(filepath, fileobject):
    """
    Save python object with pickle
    :param filepath: path for the object
    :type filepath: str
    :param fileobject: Python Object
    :type fileobject: object
    """
    # print(f"Save {os.path.basename(filepath)} at {os.path.dirname(filepath)}")
    with open(filepath, "wb") as f:
        pickle.dump(fileobject, f)


def load_object(filepath):
    """
    Load Pickle Objects
    :param filepath: Path of the file
    :type filepath: str / path
    :return: Object
    :rtype: object
    """
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj
