import Operations
from models.Configuration import ConfigurationForWeightsFile

def main():
    c = ConfigurationForWeightsFile()
    t = Operations.Trainer(c, "../best_checkpoint1.pt", "../flickr30k/flickr30k.exdir")
    

if __name__=="__main__":
    main()
