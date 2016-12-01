'''Tests for util module'''

from unittest.mock import patch, MagicMock
from memn2n import util

import numpy as np


FILE_CONTENTS = '''1 Mary got the milk.
2 John moved to the bedroom.
3 Daniel journeyed to the office.
4 John grabbed the apple there.
5 John got the football.
6 John journeyed to the garden.
7 Mary left the milk.
8 John left the football.
9 Daniel moved to the garden.
10 Daniel grabbed the football.
11 Mary moved to the hallway.
12 Mary went to the kitchen.
13 John put down the apple there.
14 John picked up the apple.
15 Sandra moved to the hallway.
16 Daniel left the football there.
17 Daniel took the football.
18 John travelled to the kitchen.
19 Daniel dropped the football.
20 John dropped the apple.
21 John grabbed the apple.
22 John went to the office.
23 Sandra went back to the bedroom.
24 Sandra took the milk.
25 John journeyed to the bathroom.
26 John travelled to the office.
27 Sandra left the milk.
28 Mary went to the bedroom.
29 Mary moved to the office.
30 John travelled to the hallway.
31 Sandra moved to the garden.
32 Mary moved to the kitchen.
33 Daniel took the football.
34 Mary journeyed to the bedroom.
35 Mary grabbed the milk there.
36 Mary discarded the milk.
37 John went to the garden.
38 John discarded the apple there.
39 Where was the apple before the bathroom? 	office	38 25 22
40 Sandra travelled to the bedroom.
41 Daniel moved to the bathroom.
42 Where was the apple before the hallway? 	office	38 30 26
43 Sandra got the milk.
44 Daniel travelled to the garden.
45 Where was the apple before the hallway? 	office	38 30 26
46 Sandra went back to the bathroom.
47 Daniel took the apple there.
48 Mary went back to the hallway.
49 Daniel went to the hallway.
50 Sandra went to the kitchen.
51 Mary journeyed to the bedroom.
52 Sandra journeyed to the hallway.
53 Daniel put down the apple.
54 Daniel put down the football there.
55 Sandra journeyed to the garden.
56 Where was the football before the garden? 	bathroom	54 44 41
57 Mary travelled to the office.
58 Sandra dropped the milk.
59 Where was the football before the garden? 	bathroom	54 44 41
1 Mary got the milk.
2 John moved to the bedroom.
3 Daniel journeyed to the office.
4 John grabbed the apple there.
5 John got the football.
6 John journeyed to the garden.
7 Mary left the milk.
8 John left the football.
9 Daniel moved to the garden.
10 Daniel grabbed the football.
11 Mary moved to the hallway.
12 Mary went to the kitchen.
13 John put down the apple there.
14 John picked up the apple.
15 Sandra moved to the hallway.
16 Daniel left the football there.
17 Daniel took the football.
18 John travelled to the kitchen.
19 Daniel dropped the football.
20 John dropped the apple.
21 John grabbed the apple.
22 John went to the office.
23 Sandra went back to the bedroom.
24 Sandra took the milk.
25 John journeyed to the bathroom.
26 John travelled to the office.
27 Sandra left the milk.
28 Mary went to the bedroom.
29 Mary moved to the office.
30 John travelled to the hallway.
31 Sandra moved to the garden.
32 Mary moved to the kitchen.
33 Daniel took the football.
34 Mary journeyed to the bedroom.
35 Mary grabbed the milk there.
36 Mary discarded the milk.
37 John went to the garden.
38 John discarded the apple there.
39 Where was the apple before the bathroom? 	office	38 25 22'''


def test_load_dataset():
    '''Test dataset load'''

    m = MagicMock(name='open')
    m().__enter__().__iter__.return_value = iter(FILE_CONTENTS.splitlines())
    with patch('memn2n.util.open', m):
        batch = list(util.load_dataset('filename'))

        assert len(batch) == 6
        sample = batch[0]
        assert len(sample) == 3
        assert len(sample[0]) == 38
        assert sample[1] == 'Where was the apple before the bathroom ?'
        assert sample[2] == 'office'


def test_memory_capacity_calc():
    m = MagicMock(name='open')
    m().__enter__().__iter__.return_value = iter(FILE_CONTENTS.splitlines())
    with patch('memn2n.util.open', m):
        dataset = list(util.load_dataset('filename'))
        capacity = util.calc_memory_capacity_for(dataset)
        assert capacity == 59 - 5


def test_sentence_length_calc():
    m = MagicMock(name='open')
    m().__enter__().__iter__.return_value = iter(FILE_CONTENTS.splitlines())
    with patch('memn2n.util.open', m):
        dataset = list(util.load_dataset('filename'))
        capacity = util.calc_sentence_length_for(dataset)
        assert capacity == 8


def test_vectorize_dataset():
    from nltk.tokenize import word_tokenize as tokenize
    m = MagicMock(name='open')
    m().__enter__().__iter__.return_value = iter(FILE_CONTENTS.splitlines())
    with patch('memn2n.util.open', m):
        dataset = list(util.load_dataset('filename'))

        memory_size, sentence_length = 20, 15
        # fake vocabulary to make sure we get correct memories
        word2idx = {
           'John': 1,
           'discarded': 2,
           'the': 3,
           'apple': 4,
           'there': 5,
           '.': 6
        }
        fcts, q, a = util.vectorize_dataset(dataset, word2idx, memory_size, sentence_length)

        assert fcts.shape == (6, memory_size, sentence_length)
        assert q.shape == (6, sentence_length)
        # last memory from particular sample
        assert np.all(fcts[-1][-1] == [1, 2, 3, 4, 5, 6] + [0, 0, 0, 0, 0, 0, 0, 0, 0])


