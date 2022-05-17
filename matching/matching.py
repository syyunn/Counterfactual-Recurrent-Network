import pandas as pd
# from strsimpy.ngram import NGram
from name_matching.name_matcher import NameMatcher

# trigram = NGram(3)

bjp = pd.read_csv('./bjp_2009_names.csv', encoding='utf-8')
allocs = pd.read_csv("./march2022_allcos.txt", sep='|')
allocs = allocs[:10]
bjp_partial = bjp[:10]
allocs['name'] = allocs['company_name']
# cleanse names

def remove_u(row):
    return row.split('\u2002')[-1]

bjp['name'] = bjp.apply(lambda row : remove_u(row['name']), axis = 1)

# company_names = ['Samsung', 'Samsung Inc.', 'Apple', 'Apple Inc']

# def get_trigram(x, y):
#     return trigram.distance(x, y)
#
# matrix = pd.DataFrame(index=bjp['name'], columns=allocs['company_name'][:10])
# df = matrix.apply(lambda x: pd.DataFrame(x).apply(lambda y: get_trigram(x.name, y.name), axis=1))


# df = pd.DataFrame(index=bjp['name'], columns = allocs['company_name'][:100])
#
# df = df.apply(get_trigram)

# initialise the name matcher
matcher = NameMatcher(
                      number_of_matches=3,
                      legal_suffixes=False,
                      common_words=False,
                      top_n=50,
                      verbose=True)

# adjust the distance metrics to use
# matcher.set_distance_metrics(
#                              bag=True,
#                              typo=True,
#                              refined_soundex=True)

# load the data to which the names should be matched
matcher.load_and_process_master_data(column='name', df_matching_data=bjp, transform=True)

# perform the name matching on the data you want matched
matches = matcher.match_names(to_be_matched=bjp_partial, column_matching='name')

if __name__ == "__main__":
    pass
