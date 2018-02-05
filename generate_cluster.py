import matplotlib.pyplot as plt, numpy as np, random, pandas as pd
import findThePapers, generate_text_list, feature_extractor
from collections import Counter, defaultdict
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.font_manager import FontProperties


def plot_clusters(num_clusters, feature_matrix,
                  cluster_data, clusters, titles,
                  plot_size=(16,8)):
    # generate random color for clusters                  
    def generate_random_color():
        color = '#%06x' % random.randint(0, 0xFFFFFF)
        return color
    # define markers for clusters    
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
    # build cosine distance matrix
    cosine_distance = 1 - cosine_similarity(feature_matrix) 
    # dimensionality reduction using MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", 
              random_state=1)
    # get coordinates of clusters in new low-dimensional space
    plot_positions = mds.fit_transform(cosine_distance)  
    x_pos, y_pos = plot_positions[:, 0], plot_positions[:, 1]
    # build cluster plotting data
    cluster_color_map = {}
    cluster_name_map = {}
    for cluster_num, cluster_details in cluster_data.items():
        # assign cluster features to unique label
        cluster_color_map[cluster_num] = generate_random_color()
        cluster_name_map[cluster_num] = ', '.join(cluster_details['key_features'][:5]).strip()
    # map each unique cluster label with its coordinates and movies
    cluster_plot_frame = pd.DataFrame({'x': x_pos,
                                       'y': y_pos,
                                       'label': clusters,
                                       'title': titles
                                        })
    grouped_plot_frame = cluster_plot_frame.groupby('label')
    # set plot figure size and axes
    fig, ax = plt.subplots(figsize=plot_size) 
    ax.margins(0.05)
    # plot each cluster using co-ordinates and movie titles
    for cluster_num, cluster_frame in grouped_plot_frame:
         marker = markers[cluster_num] if cluster_num < len(markers) \
                  else np.random.choice(markers, size=1)[0]
         ax.plot(cluster_frame['x'], cluster_frame['y'], 
                 marker=marker, linestyle='', ms=12,
                 label=cluster_name_map[cluster_num], 
                 color=cluster_color_map[cluster_num], mec='none')
         ax.set_aspect('auto')
         ax.tick_params(axis= 'x', which='both', bottom='off', top='off',        
                        labelbottom='off')
         ax.tick_params(axis= 'y', which='both', left='off', top='off',         
                        labelleft='off')
    fontP = FontProperties()
    fontP.set_size('small')    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), fancybox=True, 
              shadow=True, ncol=5, numpoints=1, prop=fontP) 
    #add labels as the film titles
    for index in range(len(cluster_plot_frame)):
        ax.text(cluster_plot_frame.ix[index]['x'], 
                cluster_plot_frame.ix[index]['y'], 
                cluster_plot_frame.ix[index]['title'], size=8)  
    # show the plot           
    plt.show() 





def get_cluster_data(clustering_obj, clusters_dict, 
                     feature_names, num_clusters,
                     topn_features=10):

    cluster_details = {}  
    # get cluster centroids
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    # get key features for each cluster
    # get movies belonging to each cluster
    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        key_features = [feature_names[index] 
                        for index 
                        in ordered_centroids[cluster_num, :topn_features]]
        cluster_details[cluster_num]['key_features'] = key_features
        cluster_details[cluster_num]['titles'] = clusters_dict[cluster_num]
    
    return cluster_details
        
       
    
def print_cluster_data(cluster_data):
    # print cluster details
    for cluster_num, cluster_details in cluster_data.items():
        print 'Cluster {} details:'.format(cluster_num)
        print '-'*20
        print 'Key features:', cluster_details['key_features']
        print 'Papers in this cluster:'
        print ', '.join(cluster_details['titles'])
        print '='*40

def affinity_propagation(feature_matrix):
    
    sim = feature_matrix * feature_matrix.T
    sim = sim.todense()
    ap = AffinityPropagation()
    ap.fit(sim)
    clusters = ap.labels_          
    return ap, clusters

def k_means(feature_matrix, num_clusters=5):
    km = KMeans(n_clusters=num_clusters, max_iter=10000)
    km.fit(feature_matrix)
    clusters = km.labels_
    return km, clusters


def clustering(files):

    files_extended = findThePapers.findThePapers(files)
    files_text = generate_text_list.generate_text_list(files_extended)
    files_tfidf_vectorizer, files_tfidf_features = feature_extractor.build_feature_matrix(files_text,
                                                        feature_type='tfidf',
                                                        ngram_range=(1, 2), 
                                                        min_df=0.24, max_df=0.85)
    
    feature_names = files_tfidf_vectorizer.get_feature_names()
    num_clusters = 5  
    # km_obj, clusters = k_means(files_tfidf_features, num_clusters)
    ap_obj, clusters = affinity_propagation(feature_matrix=files_tfidf_features)

    c = Counter(clusters)
    total_clusters = len(c)
    clusters_dict = defaultdict(lambda: [])

    for index, each in enumerate(clusters):
        i = files_extended[index].rfind('/')
        clusters_dict[each].append(files_extended[index][i+1:])

    # cluster_data =  get_cluster_data(clustering_obj=km_obj,
    #                              clusters_dict=clusters_dict,
    #                              feature_names=feature_names,
    #                              num_clusters=num_clusters,
    #                              topn_features=5)      

    cluster_data =  get_cluster_data(clustering_obj=ap_obj,
                                 clusters_dict=clusters_dict,
                                 feature_names=feature_names,
                                 num_clusters=total_clusters,
                                 topn_features=5)     

    print_cluster_data(cluster_data) 

    titles = []

    for each in files_extended:
        index = each.rfind('/')
        titles.append(each[index+1:])

    plot_clusters(num_clusters=num_clusters, 
              feature_matrix=files_tfidf_features,
              cluster_data=cluster_data, 
              clusters=clusters,
              titles = titles,
              plot_size=(16,8))  


