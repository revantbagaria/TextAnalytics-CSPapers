import pandas as pd


def display_features(features, feature_names, index_names=None):
    df = pd.DataFrame(data=features, index=index_names,
                      columns=feature_names)
    # print df
    df.to_csv("Similarities.csv")

