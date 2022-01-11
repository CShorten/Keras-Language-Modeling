# Citation to Keras Code Examples
def get_data_from_text_files(folder_name):
  def get_text_list_from_files(files):
    text_list = []
    for name in files:
        with open(name) as f:
            for line in f:
                text_list.append(line)
    return text_list
  
  pos_files = glob.glob("aclImdb/" + folder_name + "/pos/*.txt")
  pos_texts = get_text_list_from_files(pos_files)
  neg_files = glob.glob("aclImdb/" + folder_name + "/neg/*.txt")
  neg_texts = get_text_list_from_files(neg_files)
  df = pd.DataFrame(
    {
      "review": pos_texts + neg_texts,
      "sentiment": [0] * len(pos_texts) + [1] * len(neg_texts),
    }
  )
  df = df.sample(len(df)).reset_index(drop=True)
  return df
