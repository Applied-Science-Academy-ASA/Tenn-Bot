import os

import json

if __name__ == "__main__" or __name__ == "metadata":
    from features import new_features
    from Pwd import pwd
else:
    from .features import new_features
    from .Pwd import pwd

class Metadata:
    def __init__(self, trainTest=True):
        fileName = "metadata.json" if trainTest else "metadataTest.json"
        self.path = os.path.join(pwd(__file__),fileName)
        self.metadata = {}
        self.load_json()

    def load_json(self):
        if os.path.exists(self.path):
            with open(self.path,'r') as file:
                self.metadata = json.load(file)
                file.close()
        else:
            # auto create empty json
            self.save_json()
        return self.metadata
    
    def save_json(self):
        with open(self.path,'w') as file:
            json.dump(self.metadata,file)
            file.close()
        return self.metadata

    def add_features_mask(self, key, masked1, masked2,  mask):
        features = [str(x) for x in new_features(masked1, mask)]
        features += [str(x) for x in new_features(masked2, mask)]
        self.metadata[key] = ' '.join(features) #key itu index
    
    def add_features(self, key, features):
        features = [str(x) for x in features]
        self.metadata[key] = ' '.join(features) #key itu index
    
    def metadataList(self, key):
        return self.metadata[key].split(' ')

if __name__ == "__main__":
    metadata = Metadata()
    print(metadata.metadata)
    
    
