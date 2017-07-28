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

    feature_operation = None

    def __init__(self, fOp):
        if fOp not in ['create', 'scaling']:
            raise NotImplementedError('The operation {} is not implemented'.format(fOp))
        f = FeatureExtract()
        Aux.feature_operation = fOp
        self.data = f.train.append(f.test)

    def operate(self, sign, new_feature, *features):
        """
        It takes a string as an argument
        and then performs the mathematical
        operations mentioned in the arguments
        with the features passed. It can only
        support some basic arguments. Complex
        arguments are may not work properly

        The new features takes the name mentioned
        in ```self.new_feature```

        :param sign: Operation signs (+, -, *, /)
        :param new_feature: Name of the new feature
        :param features: feature names
        :return: Pandas DataFrame with the new feature
        """

        if Aux.feature_operation != 'create':
            raise RuntimeError('The Aux class object can not instantiate the '
                               'method for the {} feature operation'.format(Aux.feature_operation))

        self.data[new_feature] = self.data[features[0]]
        for each in features[1::]:
            self.data[new_feature] += self.data[each]
        return self.data[['new_feature', 'bonus', 'salary']]


if __name__ == '__main__':
    aux = Aux(fOp='create')
    print aux.operate('+', 'new_feature', 'bonus', 'salary')
