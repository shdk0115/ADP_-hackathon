def get_decompound_rules(result_df):
    keywords_list = []
    for keywords in result_df['Table_Keywords']:
        keywords_list.extend(keywords)
    decompound_rules_df = pd.DataFrame(keywords_list, columns = ['keywords'])
    print("Shape With Duplicates:", decompound_rules_df.shape)

    decompound_rules_df = decompound_rules_df.drop_duplicates()
    decompound_rules_list = decompound_rules_df['keywords'].tolist()
    print("Shape Without Duplicates:", len(decompound_rules_list))

    return decompound_rules_list
