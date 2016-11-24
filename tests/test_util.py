'''Tests for util module'''

from unittest.mock import patch, MagicMock
from memn2n import util

def test_load_dataset():
    '''Test dataset load'''
    file_contents = '''1 John travelled to the hallway.
2 Mary journeyed to the bathroom.
3 Where is John? 	hallway	1
4 Daniel went back to the bathroom.
5 John moved to the bedroom.
6 Where is Mary? 	bathroom	2
7 John went to the hallway.
8 Sandra journeyed to the kitchen.
9 Where is Sandra? 	kitchen	8
10 Sandra travelled to the hallway.
11 John went to the garden.
12 Where is Sandra? 	hallway	10
13 Sandra went back to the bathroom.
14 Sandra moved to the kitchen.
15 Where is Sandra? 	kitchen	14
1 Sandra travelled to the kitchen.
2 Sandra travelled to the hallway.
3 Where is Sandra? 	hallway	2
4 Mary went to the bathroom.
5 Sandra moved to the garden.
6 Where is Sandra? 	garden	5
7 Sandra travelled to the office.
8 Daniel journeyed to the hallway.
9 Where is Daniel? 	hallway	8
10 Daniel journeyed to the office.
11 John moved to the hallway.
12 Where is Sandra? 	office	7
13 John travelled to the bathroom.
14 John journeyed to the office.
15 Where is Daniel? 	office	10'''

    m = MagicMock(name='open')
    m().__enter__().__iter__.return_value = iter(file_contents.splitlines())
    with patch('memn2n.util.open', m):
        batch = list(util.load_dataset('filename'))

        assert len(batch) == 10
        sample = batch[0]
        assert len(sample) == 3
        assert len(sample[0]) == 2
        assert sample[1] == 'Where is John ?'
        assert sample[2] == 'hallway'

