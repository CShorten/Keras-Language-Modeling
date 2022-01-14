import random
import numpy as np

# time this

# should wrap into a generator so you don't have to keep re-loading the entire language model dataset
# probably want to load this from disk
def masked_language_model(encoded_train_texts, mask_token_id):
  # vectorize with masking
  new_texts = np.copy(encoded_train_texts)
  rnd_idxs = list(range(256))
  labels = []

  for i, seq in enumerate(new_texts):
    mask_idx = random.choice(rnd_idxs)
    labels.append(new_texts[i][mask_idx])
    new_texts[i][mask_idx] = mask_token_id

  return new_texts, np.array(labels)
