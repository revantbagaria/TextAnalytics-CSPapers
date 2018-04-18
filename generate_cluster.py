
import matplotlib.pyplot as plt, numpy as np, random, pandas as pd
plt.style.use('ggplot')
import findThePapers, generate_text_list, feature_extractor
from generate_titles import generate_titles
from collections import Counter, defaultdict
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.font_manager import FontProperties
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import ward, dendrogram



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
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), fancybox=True, 
              shadow=True, ncol=5, numpoints=1, prop=fontP, fontsize='xx-small') 
    #add labels as the film titles
    for index in range(len(cluster_plot_frame)):
        ax.text(cluster_plot_frame.ix[index]['x'], 
                cluster_plot_frame.ix[index]['y'], 
                cluster_plot_frame.ix[index]['title'], size=8)  
    # show the plot
    plt.title("Clustering")  
    plt.savefig('SC_aff.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.savefig('foo2.png')   
    # plt.tight_layout(pad=7)  
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

def affinity_propagation(feature_matrix, titles, feature_names):
    
    sim = feature_matrix * feature_matrix.T
    sim = sim.todense()
    ap = AffinityPropagation()
    ap.fit(sim)
    clusters = ap.labels_  

    c = Counter(clusters)
    num_clusters = len(c)

    clusters_dict = defaultdict(lambda: [])

    for index, each in enumerate(clusters):
        clusters_dict[each].append(titles[index])

    cluster_data =  get_cluster_data(clustering_obj=ap,
                                 clusters_dict=clusters_dict,
                                 feature_names=feature_names,
                                 num_clusters=num_clusters,
                                 topn_features=5)   

    print_cluster_data(cluster_data) 

    plot_clusters(num_clusters=num_clusters, 
              feature_matrix=feature_matrix,
              cluster_data=cluster_data, 
              clusters=clusters,
              titles = titles,
              plot_size=(16,8))

def k_means(feature_matrix, titles, feature_names, num_clusters=5):

    km = KMeans(n_clusters=num_clusters, max_iter=10000) # max_iter defines the total number of iterations before convergence (incase it takes infinite no of iterations inorder to converge)
    km.fit(feature_matrix)
    clusters = km.labels_
    clusters_dict = defaultdict(lambda: [])

    for index, each in enumerate(clusters):
        clusters_dict[each].append(titles[index])

    cluster_data =  get_cluster_data(clustering_obj=km,
                         clusters_dict=clusters_dict,
                         feature_names=feature_names,
                         num_clusters=num_clusters,
                         topn_features=5)

    print_cluster_data(cluster_data) 

    plot_clusters(num_clusters=num_clusters, 
              feature_matrix=feature_matrix,
              cluster_data=cluster_data, 
              clusters=clusters,
              titles = titles,
              plot_size=(10,5))  

def dbscan_clustering(feature_matrix, titles, min_samples=2, eps=0.3):
    dbscan = DBSCAN(min_samples, eps)
    dbscan.fit(feature_matrix)
    clusters = dbscan.labels_
    clusters_dict = defaultdict(lambda: [])

    for index, each in enumerate(clusters):
        clusters_dict[each].append(titles[index])

    for each in clusters_dict.keys():
        print("Cluster {}: ".format(each))
        print(clusters_dict[each])


def hdbscan_clustering(feature_matrix, titles, min_cluster_size=2):
    hdb_obj = hdbscan.HDBSCAN(min_cluster_size)
    hdb_obj.fit(feature_matrix)
    clusters = hdb_obj.labels_
    clusters_dict = defaultdict(lambda: [])

    for index, each in enumerate(clusters):
        clusters_dict[each].append(titles[index])

    for each in clusters_dict.keys():
        print("Cluster {}: ".format(each))
        print(clusters_dict[each])

    # cluster_data =  get_cluster_data(clustering_obj=hdb_obj,
    #                              clusters_dict=clusters_dict,
    #                              feature_names=feature_names,
    #                              num_clusters=num_clusters,
    #                              topn_features=5) 


def plot_hierarchical_clusters(linkage_matrix, titles, figure_size=(8,12)):
    # set size
    fig, ax = plt.subplots(figsize=figure_size) 
    # plot dendrogram
    ax = dendrogram(linkage_matrix, orientation="left", labels=titles)
    plt.tick_params(axis= 'x',   
                    which='both',  
                    bottom='off',
                    top='off',
                    labelbottom='off')
    plt.tight_layout()
    plt.savefig('ward_hierachical_clusters.png', dpi=200)


def ward_hierarchical_clustering(feature_matrix, titles):
    
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    # build ward's linkage matrix   
    linkage_matrix = ward(cosine_distance)
    # plot the dendrogram
    plot_hierarchical_clusters(linkage_matrix=linkage_matrix,
                               titles=titles,
                               figure_size=(8,10))


def clustering(files, files_append, knn_clustering, affinity_prop, db, hdb, ward):

    if not files:
        files = files_append   

    files_extended = findThePapers.findThePapers(files)
    titles = generate_titles(files_extended)    
    files_text = generate_text_list.generate_text_list(files_extended)
    files_tfidf_vectorizer, files_tfidf_features = feature_extractor.build_feature_matrix(files_text,
                                                        feature_type='tfidf',
                                                        ngram_range=(1, 2), 
                                                        min_df=0.24, max_df=0.85)
    feature_names = files_tfidf_vectorizer.get_feature_names()

    # for each in files_extended:
    #   index = each.rfind('/')
    #   titles.append(each[index+1:])

    if knn_clustering:
        k_means(files_tfidf_features.toarray(), titles, feature_names, int(knn_clustering))

    if affinity_prop:
        affinity_propagation(files_tfidf_features, titles, feature_names)

    if db:
        min_samples, eps = int(db[0]), float(db[1])
        dbscan_clustering(files_tfidf_features, titles, min_samples, eps)

    if hdb:
        hdbscan_clustering(files_tfidf_features, titles, int(hdb))

    if ward:
        ward_hierarchical_clustering(files_tfidf_features, titles)

    # hdb_obj = hdbscan.HDBSCAN(min_cluster_size=2)
    # clusters = hdb_obj.fit(files_tfidf_features)
    # print(clusters.labels_)

    # dbscan = DBSCAN(min_samples=2, eps=0.3)
    # dbscan.fit(files_tfidf_features)
    # print(dbscan.labels_)


    # c = Counter(clusters)
    # total_clusters = len(c)

  

    # # cluster_data =  get_cluster_data(clustering_obj=hdb_obj,
    # #                              clusters_dict=clusters_dict,
    # #                              feature_names=feature_names,
    # #                              num_clusters=num_clusters,
    # #                              topn_features=5)  

    # print_cluster_data(cluster_data) 

    # plot_clusters(num_clusters=num_clusters, 
    #           feature_matrix=files_tfidf_features.toarray(),
    #           cluster_data=cluster_data, 
    #           clusters=clusters,
    #           titles = titles,
    #           plot_size=(16,8))  


    # a = files_tfidf_features.toarray()
    # pca = PCA(n_components=2).fit(a)
    # pca_2d = pca.transform(a)

    # for i in range(0, pca_2d.shape[0]):
    #   if clusters.labels_[i] == 0:
    #     c1 = plt.scatter(pca_2d[i,0], pca_2d[i,1],c='r', marker='+')
    #   elif clusters.labels_[i] == 1:
    #     c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
    #   elif clusters.labels_[i] == -1:
    #     c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')
    # plt.legend([c1,c2,c3], ['Cluster 1', 'Cluster 2', 'Noise'])
    # plt.title('HDBSCAN finds 2 clusters and noise')
    # plt.show()


