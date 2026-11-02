"""Exploratory label-permutation test on a recorded pairwise cosine matrix."""
import argparse
import itertools
import json
import statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument('input',type=Path); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    rows=json.loads(args.input.read_text()); nodes=sorted({tuple(z['a']) for z in rows}|{tuple(z['b']) for z in rows})
    n=len(nodes); index={v:i for i,v in enumerate(nodes)}; matrix=[[1.0]*n for _ in range(n)]
    for row in rows:
        a=index[tuple(row['a'])]; b=index[tuple(row['b'])]; matrix[a][b]=matrix[b][a]=row['cosine']
    actual=tuple(i for i,node in enumerate(nodes) if node[1].startswith('official_'))
    if len(actual)*2 != n or len(rows)!=n*(n-1)//2: raise ValueError('expected complete balanced pair matrix')
    def score(a):
        a=set(a); b=set(range(n))-a
        return (statistics.mean(matrix[i][j] for i,j in itertools.combinations(a,2))+
                statistics.mean(matrix[i][j] for i,j in itertools.combinations(b,2)))/2 - statistics.mean(matrix[i][j] for i in a for j in b)
    observed=score(actual); values=[score(c) for c in itertools.combinations(range(n),len(actual))]
    result={'n_runs':n,'observed_within_minus_cross':observed,'one_sided_p':sum(v>=observed for v in values)/len(values),
            'permutations':len(values),'interpretation':'exploratory label clustering, not an equivalence test'}
    with args.output.open('x') as f: json.dump(result,f,indent=2)
    print(result)

if __name__=='__main__': main()
