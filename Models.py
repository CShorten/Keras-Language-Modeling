def bert_module(query, key, value, embed_dims, num_heads, ff_dim):
    # Multi headed self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim= embed_dims // num_heads,
    )(query, key, value)
    attention_output = layers.Dropout(0.1)(
        attention_output
    )
    attention_output = layers.LayerNormalization(
        epsilon=1e-6,
    )(query + attention_output)

    # Feed-forward layer
    ffn = keras.Sequential(
        [
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dims),
        ],
    )
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.1)(
        ffn_output
    )
    sequence_output = layers.LayerNormalization(
        epsilon=1e-6
    )(attention_output + ffn_output)
    return sequence_output


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc
  
def get_BERT(config):
  inputs = layers.Input((256,), dtype=tf.int64)

  word_embeddings = layers.Embedding(
        config.VOCAB_SIZE, config.EMBED_DIM, name="word_embedding")(inputs)
  position_embeddings = layers.Embedding(
        input_dim=config.MAX_LEN,
        output_dim=config.EMBED_DIM,
        weights=[get_pos_encoding_matrix(config.MAX_LEN, config.EMBED_DIM)],
        name="position_embedding",
    )(tf.range(start=0, limit=config.MAX_LEN, delta=1))
  embeddings = word_embeddings + position_embeddings

  encoder_output = embeddings

  encoder_output = bert_module(encoder_output, encoder_output, encoder_output, 256, 8, 256)
  encoder_output = bert_module(encoder_output, encoder_output, encoder_output, 256, 8, 256)

  # move to hardcoded loop
  encoder_output = layers.AveragePooling1D()(encoder_output) #256 --> 128

  encoder_output = layers.AveragePooling1D()(encoder_output) #64

  encoder_output = bert_module(encoder_output, encoder_output, encoder_output, 256, 8, 256)
  encoder_output = bert_module(encoder_output, encoder_output, encoder_output, 256, 8, 256)

  encoder_output = layers.AveragePooling1D()(encoder_output) #32
  encoder_output = layers.AveragePooling1D()(encoder_output) #16

  encoder_output = bert_module(encoder_output, encoder_output, encoder_output, 256, 8, 256)
  encoder_output = bert_module(encoder_output, encoder_output, encoder_output, 256, 8, 256)

  encoder_output = layers.AveragePooling1D()(encoder_output) #8
  encoder_output = layers.AveragePooling1D()(encoder_output) #4
  encoder_output = layers.AveragePooling1D()(encoder_output) #2
  encoder_output = layers.AveragePooling1D(name="vector_representation")(encoder_output) #1

  encoder_output = layers.Flatten()(encoder_output)

  #encoder_output = layers.Dense(5096, activation="relu")(encoder_output)

  mlm_output = layers.Dense(config.VOCAB_SIZE, activation="softmax")(encoder_output)

  model = keras.Model(inputs, mlm_output)
  return model
