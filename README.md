some notes:

# BPE optimization

From 3h -> 13s

- parallelized pretokenization with <|endoftext|> landmarks, not really significant 
- inverted index map (map from a bytepair to the set of pretokens containing it). before i scanned every pretoken each merge to find occurences of the pair to merge, now just iterates on the set
- maxheap for argmax instead of recounting each iteration (made argmax time negligible, but increased merge timing because of all the excess pushes)
- batched pushes per merge, not per change,

final timing:

===== timings (4811 merges, 627486 distinct pretokens) =====
  pretokenize            5.332s   40.9%
  initial_pair_count     5.052s   38.8%
  merge                  2.651s   20.3%
  argmax                 0.000s    0.0%
  total                 13.036s
  per merge: 0.551 ms
