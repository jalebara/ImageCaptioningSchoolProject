import Operations

def main():
    evaluator = Operations.Evaluator("../best_checkpoint1_fixed.pt", "../flickr30k/flickr30k.exdir", smoke_test=True, fast_test=False)
    print(evaluator.evaluate())
    

if __name__=="__main__":
    main()
