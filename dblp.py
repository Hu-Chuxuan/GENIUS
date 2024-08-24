
import torch
import json
from itertools import permutations

filename = "dblp_papers_v11.txt" # can be downloaded from https://www.aminer.org/citation, V11
def data_preprocess(filename):
    authordict = {}
    fosdict = {}

    i = 0
    j = 0
    c = 0

    f = open(filename)

    line = f.readline()

    while line and c < 3000:
        line = json.loads(line)
        team = []
        if 'authors' in line.keys():
            for author in line['authors']:
                team.append(author['name'])
        team = list(set(team))
        if len(team) <= 2:
            line = f.readline()
            continue
        for author in line['authors']:
            if author['name'] in authordict.keys():
                continue
            else:
                authordict[author['name']] = i
                i += 1
        if 'fos' in line.keys():
            for fos in line['fos']:
                if fos['name'] in fosdict.keys():
                    continue
                else:
                    fosdict[fos['name']] = j
                    j += 1
        c += 1
        line = f.readline()
        

    f.close() 
    
    x = torch.zeros(len(authordict.keys()),len(fosdict.keys()))
    adj = torch.zeros(len(authordict.keys()),len(authordict.keys()))

    papers = torch.zeros(3000,len(fosdict.keys()))

    teams = []

    f = open(filename)

    line = f.readline()

    c = 0

    while c < 3000:
        line = json.loads(line)
        team = []
        if 'authors' in line.keys():
            for author in line['authors']:
                team.append(author['name'])
        team = list(set(team))
        if len(team) <= 2:
            line = f.readline()
            continue    

        team = []
        if 'fos' in line.keys():
            for fos in line['fos']:
                ind = fosdict[fos['name']]
                papers[c][ind] += 1

        if 'authors' in line.keys():
            for author in line['authors']:
                ind = authordict[author['name']]
                x[ind] += papers[c]
                team.append(ind)
                
        teams.append(list(set(team)))
        c += 1
        
        line = f.readline()

    f.close() 
    for paper in teams:
        perm = permutations(paper, 2)
        for pair in list(perm):
            adj[pair[0]][pair[1]] += 1

    return adj, x, teams