
class bin:

    def __init__(self, id: int) -> None:

        self.bin_id: int = id
        self.samples: list(tuple) = []
    
    @staticmethod
    def check_values(prob, label):

        if isinstance(prob) != float :
            raise ValueError(f'variable prob contains value of incorrect datatype.\
                Expected float found {isinstance(prob)}')
        
        if not 0<=prob<=1:
            raise ValueError(f'Value out of Bound. Expected value between 0 and 1.')
        
        if isinstance(label) != int:
            raise ValueError(f'variable label contains value of incorrect datatype.\
                Expected float found {isinstance(label)}')
        

    def add_sample(self, prob: float, label: int) -> None:

        self.check_values(prob, label)
        
        self.samples.append((prob, label))

class HistogramBinningCalibrator:

    def __init__(self, probs, labels, partition_scheme='equal-impression'):

        self.probs = probs
        self.labels = labels
        self.partition_scheme = partition_scheme

        # check for errors
        self.check_values(probs, labels)
        self.samples = [(prob, label) for prob, label in zip(probs, labels)]

        
    
    @staticmethod
    def check_values(prob, label):

        if len(prob) != len(label) :
            raise ValueError(f'Size mismatch. prob contains {len(prob)} elements \
                label contains {len(label)} element')
    
    def fit(self):

        # create partition
        sorted_samples = sorted( self.samples, key=lambda x: x[0])
        partitioned_bins = get_partition(sorted_samples, self.partition_scheme)
        


        # learn bin wise statistics


