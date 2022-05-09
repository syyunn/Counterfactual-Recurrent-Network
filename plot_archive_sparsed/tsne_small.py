import random
import pickle


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

with open('./training_br_states_small_binary.pickle', "rb") as handle:
    training_br_states = pickle.load(handle)

with open('./training_processed_small_binary.pickle', "rb") as handle:
    training_processed = pickle.load(handle)


t = random.randrange(0, 65)

for t in range(1, 66):
    print(t)
    brs = training_br_states[:,t,:]
    treatments = training_processed['current_treatments'][:,t,:]
    # treatments[treatments > 0] = 1
    # treatments[treatments <= 0] = 0
    cvrs = training_processed['current_covariates'][:,t,:]

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results_brs = tsne.fit_transform(brs)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results_cvrs = tsne.fit_transform(cvrs)

    df = pd.DataFrame(columns=['tsne-2d-one-br', 'tsne-2d-two-br', 'treatment', 'tsne-2d-one-cvrs', 'tsne-2d-two-cvrs'])
    df['tsne-2d-one-br'] = tsne_results_brs[:, 0]
    df['tsne-2d-two-br'] = tsne_results_brs[:, 1]
    df['tsne-2d-one-cvrs'] = tsne_results_cvrs[:, 0]
    df['tsne-2d-two-cvrs'] = tsne_results_cvrs[:, 1]
    df['treatment'] = treatments[:,0]

    # plt.figure(figsize=(16,10))
    fig, ax = plt.subplots(1,2)

    sns.scatterplot(
        x="tsne-2d-one-br", y="tsne-2d-two-br",
        hue="treatment",
        # palette=sns.color_palette("hls", 2),
        palette='seismic',
        data=df,
        # legend="full",
        alpha=0.3,
        ax=ax[1]
    )

    sns.scatterplot(
        x="tsne-2d-one-cvrs", y="tsne-2d-two-cvrs",
        hue="treatment",
        # palette=sns.color_palette("hls", 2),
        palette='seismic',
        data=df,
        # legend="full",
        alpha=0.3,
        ax=ax[0]
    )

    plt.show()

if __name__ == "__main__":
    pass
