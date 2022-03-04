import Operations
from models.Configuration import ConfigurationForWeightsFile

def main():
    t = Operations.Trainer("../best_checkpoint1_fixed.pt", "../flickr30k/flickr30k.exdir", smoke_test=True, fast_test=False)
    t.train_one_epoch()
    

if __name__=="__main__":
    main()
