import pickle

def save(file, name):
    with open(name, 'wb') as f:
        pickle.dump(file, f)

def load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)