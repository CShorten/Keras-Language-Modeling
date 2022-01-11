from tensorflow.keras.layers import TextVectorization
import tensorflow as tf

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}~"), ""
    )
  
def get_vectorize_layer(texts, vocab_size, max_seq, special_tokens=["[MASK], [start], [end]"]):
    """Build Text vectorization layer

    Args:
      texts (list): List of string i.e input texts
      vocab_size (int): vocab size
      max_seq (int): Maximum sequence lenght.
      special_tokens (list, optional): List of special tokens. Defaults to ['[MASK]'].

    Returns:
        layers.Layer: Return TextVectorization Keras Layer
    """
    vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        standardize=custom_standardization,
        output_sequence_length=max_seq,
    )
    vectorize_layer.adapt(texts)

    # Insert mask token in vocabulary
    vocab = vectorize_layer.get_vocabulary()
    vocab = vocab[2 : vocab_size - len(special_tokens)] + ["[mask]"]
    vocab = vocab[2 : vocab_size - len(special_tokens)] + ["[start]"]
    vocab = vocab[2 : vocab_size - len(special_tokens)] + ["[end]"]


    vectorize_layer.set_vocabulary(vocab)
    return vectorize_layer
  
def encode(dataset, vocab_size, max_len, special_tokens):
  vectorize_layer = get_vectorize_layer(
    dataset.review.values.tolist(), vocab_size, max_len, special_tokens
  )
  special_ids = []
  for i in range(len(special_tokens)):
    special_ids.append(vectorize_layer([special_tokens[i]]).numpy()[0][0])
  
  encoded_dataset = vectorize_layer(dataset)
  return encoded_dataset.numpy(), special_ids
