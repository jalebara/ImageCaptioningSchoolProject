from models.meshed_memory import MeshedMemoryTransformer
from models.Configuration import Configuration
import torch

class BeamSearch(object):
    def __init__(self, model: MeshedMemoryTransformer, beam_size: int, config: Configuration):
        self.model = MeshedMemoryTransformer
        self.config = config
        self.beam_size = beam_size

    def apply(self, data):
        # Stores our beam_size best sequences, initialized as pad tokens
        best_sequences = torch.full([self.beam_size, self.config["max_sequence_length"]], self.config["pad_token"])
        # Make the first column all start tokens
        best_sequences[:, 0] = torch.full([self.beam_size], self.config["start_token"])
        # Stores the current scores of the best sequences, initialized to 0
        best_scores = torch.full([self.beam_size], 0)
        
        for i in range(1,self.config["max_sequence_length"]):
            scores = torch.empty([self.beam_size*self.beam_size])
            best_words = torch.empty([self.beam_size*self.beam_size])
            for j in range(0, self.beam_size):
                # Pass the current jth best sequence into the transformer
                temp_seq = self.model(data, best_sequences[j, :])
                # temp_seq is a max_sequence_length x vocab_size array of scores
                # We care about ith row in temp_seq
                column_under_consideration = temp_seq[:, i]
                temp_scores, temp_best_words = torch.topk(column_under_consideration, self.beam_size)
                # Compute log probabilities of the scores, add to the previous score
                temp_scores = torch.log(temp_scores) + best_scores[j]

                # reshape temp_scores and temp_best_words to 1 x beam_size
                temp_scores = temp_scores.unsqueeze(0)
                temp_best_words = temp_best_words.unsqueeze(0)

                # Put the temp_scores and temp_best_words into the scores and best_words tensor
                scores[range(self.beam_size*j, self.beam_size*j+self.beam_size)] = temp_scores
                best_words[range(self.beam_size*j, self.beam_size*j+self.beam_size)] = temp_best_words
            # We now have the beam_size^2 best successor words in best_words and the corresponding scores in scores
            
            # Find the top beam_size scores
            best_candidate_words, best_candidate_idxs = torch.topk(scores)
            
            # Create "new" best sequences
            new_best_sequences = torch.full([self.beam_size, self.config["max_sequence_length"]], self.config["pad_token"])
            k = 0
            for best_candidate_word, best_candidate_idx in best_candidate_words, best_candidate_idxs:
                # which sequence does the candidate word belong to?
                membership_idx = torch.div(best_candidate_idx, self.beam_size, rounding_mode='floor')
                new_best_sequences[k, range(0,i)] = best_sequences[membership_idx, range(0,i)]
                new_best_sequences[k, i+1] = best_candidate_word
                best_scores[k] = scores[best_candidate_idx]
                k += 1

            # Update best_sequences
            best_sequences = new_best_sequences
