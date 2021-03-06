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

    def __init__(self, fOp, fObj):
        if fOp not in ['create', 'scaling']:
            raise NotImplementedError('The operation {} is not implemented'.format(fOp))
        # f = FeatureExtract()
        Aux.feature_operation = fOp
        self.feature_object = fObj
        # print self.feature_object
        self.data = self.feature_object.df

    def operate(self, sign, new_feature, features, remove=True):
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
        :param remove: remove the old features
        :return: Pandas DataFrame with the new feature
        """

        if Aux.feature_operation != 'create':
            raise RuntimeError('The Aux class object can not instantiate the '
                               'method for the {} feature operation'.format(Aux.feature_operation))

        self.data[new_feature] = self.data[features[0]]
        for each in features[1::]:
            if sign == '+':
                self.data[new_feature] += self.data[each]
            elif sign == '-':
                self.data[new_feature] -= self.data[each]
            elif sign == '*':
                self.data[new_feature] *= self.data[each]
            elif sign == '/':
                self.data[new_feature] /= self.data[each]

        # Adding the new feature
        FeatureExtract.featureCols.append(new_feature)
        # Removing the old features
        if remove:
            for each in features:
                FeatureExtract.featureCols.remove(each)
        self.data.fillna(0.0, inplace=True)
        # This is done to keep target at the end
        target = self.data[FeatureExtract.targetCol]
        self.data.drop([FeatureExtract.targetCol], axis=1, inplace=True)
        self.data[FeatureExtract.targetCol] = target
        self.feature_object.df = self.data


if __name__ == '__main__':
    aux = Aux(fOp='create')
    print aux.operate('+', 'new_feature', 'bonus', 'salary')
