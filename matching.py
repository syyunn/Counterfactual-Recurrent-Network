import pandas as pd
from nltk import ngrams

pd.read_Csv('./B')

company_names = ['Samsung', 'Samsung Inc.', 'Apple', 'Apple Inc']


def get_trigram(cell):
    cell.index
    return cell

df = pd.DataFrame(index=company_names, columns = company_names)

df = df.apply(get_trigram)

if __name__ == "__main__":
    pass
