"""
This file contains some special methods that perform
some more feature operations like

1. Feature Scaling
2. Feature Creation
"""

from feature_loader import FeatureExtract

# f = FeatureExtract()
# print f.train.head(1)


class Aux:

    def __init__(self, operation, **kwargs):
        if operation not in ['create', 'scale']:
            raise NotImplementedError('The mentioned feature operation {} is not implemented'.format(operation))
        if operation.lower() == 'create':
            self.new_feature = kwargs.get('new_feature')

    def operate(self, argument, **kwargs):
        """
        It takes a string as an argument
        and then performs the mathematical
        operations mentioned in the arguments
        with the features passed. It can only
        support some basic arguments. Complex
        arguments are may not work properly

        For Example:
            argument: f1 / f2 + f3
            kwargs: f1 = 'bonus'
                    f2 = 'salary'
                    f3 = 'total_payments'

        The new features takes the name mentioned
        in ```self.new_feature```

        :param argument: Operation arguments (str)
        :param kwargs: feature names
        :return: Pandas Series of new feature
        """