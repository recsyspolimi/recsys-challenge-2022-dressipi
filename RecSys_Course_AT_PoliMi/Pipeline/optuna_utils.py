

class Space:

    IDENTIFIER_SPACE = "SPACE_CLASS"

    def __init__(self):
        raise NotImplemented

    def set_trial(self, trial):
        self.trial = trial

    def set_name(self, name):
        self.name = name

    def suggest(self):
        raise NotImplementedError


class Range(Space):

    def __init__(self, low=0, high=1, prior='uniform'):
        self.low = low
        self.high = high
        self.prior = prior


class Categorical(Space):

    def __init__(self, params):
        if type(params) != list:
            self.params = list(params)

        self.params = params

    def suggest(self):
        return self.trial.suggest_categorical(self.name, self.params)


class Integer(Range):

    def __init__(self, low=0, high=1, prior='uniform', step=1):
        super(Integer, self).__init__(low, high, prior)
        self.step = step

    def suggest(self):
        return self.trial.suggest_int(self.name, self.low, self.high, self.step)


class Real(Range):

    def __init__(self, low=0, high=1, prior='uniform', step=None):
        super(Real, self).__init__(low, high, prior)
        self.log = True if self.prior == 'log-uniform' else False
        self.step = None if self.log is True else step

    def suggest(self):
        return self.trial.suggest_float(self.name, self.low, self.high, step=self.step, log=self.log)


def suggest(trial, param_dict):
    sampled = {}

    for param, val in param_dict.items():

        if hasattr(val, "IDENTIFIER_SPACE"):
            val.set_name(param)
            val.set_trial(trial)
            sampled[param] = val.suggest()
        else:
            sampled[param] = val

    return sampled