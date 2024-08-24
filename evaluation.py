from gklearn.kernels import ShortestPath, PathUpToH, Marginalized
import networkx as nx

def numpy_to_graph(A,node_features):
    '''Convert numpy arrays to graph

    Parameters
    ----------
    A : mxm array
        Adjacency matrix
    type_graph : str
        'dgl' or 'nx'
    node_features : dict
        Optional, dictionary with key=feature name, value=list of size m
        Allows user to specify node features

    Returns

    -------
    Graph of 'type_graph' specification
    '''
    
    G = nx.from_numpy_array(A)
    
    for n in G.nodes():
        for i in range(len(node_features[n])):
            G.nodes[n][i] = node_features[n][i]
                
    return G

def Criterion(team_orig,team_new, x, adj):
    ged = 0
    wl = 0
    sp = 0
    p = 0
    m = 0
    for i in range(len(team_orig)):
        A0 = adj.take(team_orig[i],0).take(team_orig[i],1)
        X0 = x.take(team_orig[i],0)
        G0 = numpy_to_graph(A0,X0)

        A1 = adj.take(team_new[i],0).take(team_new[i],1)
        X1 = x.take(team_new[i],0)
        G1 = numpy_to_graph(A1,X1)

        tmp = nx.graph_edit_distance(G0, G1, node_match=lambda x,y : x==y)
        ged += tmp
        print("The", i, "th sample has GED = ", tmp)

        G = ShortestPath()
        tmp0 = G._compute_single_kernel_series(G0, G0)
        tmp = G._compute_single_kernel_series(G0, G1)
        _ = abs(tmp-tmp0)/tmp0
        sp += _
        print("The", i, "th sample has ShortestPath = ", tmp)
        print("The", i, "th sample has ShortestPath Base = ", tmp0)
        print("The", i, "th sample has ShortestPath diff = ", _)

        G = PathUpToH()
        tmp = G._compute_single_kernel_series(G0, G1)
        tmp0 = G._compute_single_kernel_series(G0, G0)
        _ = abs(tmp-tmp0)/tmp0
        p += _
        print("The", i, "th sample has PathUpToH = ", tmp)
        print("The", i, "th sample has PathUpToH Base = ", tmp0)
        print("The", i, "th sample has PathUpToH diff = ", _)

        G = Marginalized()
        tmp = G._compute_single_kernel_series(G0, G1)
        tmp0 = G._compute_single_kernel_series(G0, G0)
        _ = abs(tmp-tmp0)/tmp0
        m += _
        print("The", i, "th sample has Marginalized = ", tmp)
        print("The", i, "th sample has Marginalized Base = ", tmp0)
        print("The", i, "th sample has Marginalized diff = ", _)

    print("ged = ", ged)
    print("wl = ", wl)
    print("sp = ", sp)
    print("p = ", p)
    print("m = ", m)