#!/usr/bin/env python
"""
This module provides a class for describing pulses by an analytical formula
"""
from __future__ import print_function, division, absolute_import, \
                       unicode_literals
import re
import sys
import numpy as np
from QDYN.pulse import Pulse, pulse_tgrid, carrier, blackman
import json
import inspect


class NumpyAwareJSONEncoder(json.JSONEncoder):
    """JSON Encoder than can handle 1D real numpy arrays by converting them to
    to a special object. The result can be decoded using the
    NumpyAwaraJSONDecoder to recover the numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return {'type': 'np.'+obj.dtype.name, 'vals' :obj.tolist()}
        return json.JSONEncoder.default(self, obj)


class SimpleNumpyAwareJSONEncoder(json.JSONEncoder):
    """JSON Encoder than can handle 1D real numpy arrays by converting them to
    a list. Note that this does NOT allow to recover the original numpy array
    from the JSON data"""
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class NumpyAwareJSONDecoder(json.JSONDecoder):
    """Decode JSON data that hs been encoded with NumpyAwareJSONEncoder"""
    def __init__(self, *args, **kargs):
        json.JSONDecoder.__init__(self, object_hook=self.dict_to_object,
                                  *args, **kargs)
    def dict_to_object(self, d):
        inst = d
        if (len(d) == 2) and ('type' in d) and ('vals' in d):
            type = d['type']
            vals = d['vals']
            if type.startswith("np."):
                dtype = type[3:]
            inst = np.array(vals, dtype=dtype)
        return inst


class AnalyticalPulse(object):
    """Representation of a pulse determined by an analtyical formula"""
    _formulas = {} # formula name => formula callable, see `register_formula()`
    _allowed_args = {}  # formula name => allowed arguments
    _required_args = {} # formula name => required arguments

    @classmethod
    def register_formula(cls, name, formula):
        """Register a new analytical formula

        Parameters
        ----------
        name: str
            Label for the formula
        formula: callable
            callable that takes an tgrid numpy array and an arbitrary number of
            (keyword) arguments and returns a numpy array of amplitude values
        """
        argspec = inspect.getargspec(formula)
        if len(argspec.args) < 1:
            raise ValueError("formula has zero arguments, must take at least "
                             "a tgrid parameter")
        cls._formulas[name] = formula
        cls._allowed_args[name] = argspec.args[1:]
        n_opt = 0
        if argspec.defaults is not None:
            n_opt = len(argspec.defaults)
        cls._required_args[name] = argspec.args[1:-n_opt]

    def __init__(self, formula, T, nt, parameters, t0=0.0, time_unit='au',
        ampl_unit='au', freq_unit=None, mode=None):
        """Instantiate a new analytical pulse"""
        if not formula in self._formulas:
            raise ValueError("Unkown formula '%s'" % formula)
        self._formula = formula
        self.parameters = parameters
        self._check_parameters()
        self.t0 = t0
        self.nt = nt
        self.T = T
        self.time_unit = time_unit
        self.ampl_unit = ampl_unit
        self.freq_unit = freq_unit
        self.mode = mode

    def array_to_parameters(self, array, keys=None):
        """
        Unpack the given array (numpy array or regular list) into the pulse
        parameters. This is especially useful for optimizing parameters with
        the `scipy.optimize.mimimize` routine.

        For each key, set the value of the `parameters[key]` attribute by
        popping values from the beginning of the array. If `parameters[key]` is
        an array, pop repeatedly to set every value.

        If keys is not given, all parameter keys are used, in sorted order. The
        array must contain exactly enough parameters, otherwise an IndexError
        is raised.
        """
        if keys is None:
            keys = sorted(self.parameters.keys())
        array = list(array)
        for key in keys:
            if np.isscalar(self.parameters[key]):
                self.parameters[key] = array.pop(0)
            else:
                for i in range(len(self.parameters[key])):
                    self.parameters[key][i] = array.pop(0)
        if len(array) > 0:
            raise IndexError("not all values in array placed in parameters")

    def parameters_to_array(self, keys):
        """Inverse method to `array_to_paramters`. Returns the "packed"
        parameter values for the given keys as a numpy array"""
        result = []
        if keys is None:
            keys = sorted(self.parameters.keys())
        for key in keys:
            if np.isscalar(self.parameters[key]):
                result.append(self.parameters[key])
            else:
                for i in range(len(self.parameters[key])):
                    result.append(self.parameters[key][i])
        return np.array(result)

    def _check_parameters(self):
        """Raise a ValueError if self.parameters is missing any required
        parameters for the current formula. Also raise ValueError is
        self.parameters contains any extra parameters"""
        formula = self._formula
        for arg in self._required_args[formula]:
            if not arg in self.parameters:
                raise ValueError(('Required parameter "%s" for formula "%s" '
                                  'not in parameters %s')%(arg, formula,
                                  self.parameters))
        for arg in self.parameters:
            if not arg in self._allowed_args[formula]:
                raise ValueError(('Parameter "%s" does not exist in formula '
                                  '"%s"')%(arg, formula))

    @property
    def formula_name(self):
        """Name of the analytical formula that is used"""
        return self._formula

    @property
    def evaluate_formula(self):
        """The callable that numerically evaluates the used formula"""
        return self._formulas[self._formula]

    def to_json(self, pretty=True):
        """Return a json representation of the pulse"""
        self._check_parameters()
        json_opts = {'indent': None, 'separators':(',',':'), 'sort_keys': True}
        if pretty:
            json_opts = {'indent': 2, 'separators':(',',': '),
                         'sort_keys': True}
        attributes = self.__dict__.copy()
        attributes['formula'] = attributes.pop('_formula')
        return json.dumps(attributes, cls=NumpyAwareJSONEncoder,
                          **json_opts)

    def __str__(self):
        """Return string representation (JSON)"""
        return self.to_json(pretty=True)

    def write(self, filename, pretty=True):
        """Write the analytica pulse to the given filename as a json data
        structure"""
        with open(filename, 'w') as out_fh:
            out_fh.write(self.to_json(pretty=pretty))

    @property
    def header(self):
        """Single line summarizing the pulse. Suitable for preample for
        numerical pulse"""
        result = '# Formula "%s"' % self._formula
        if len(self.parameters) > 0:
            result += ' with '
            json_opts = {'indent': None, 'separators':(', ',': '),
                        'sort_keys': True}
            json_str = json.dumps(self.parameters,
                                  cls=SimpleNumpyAwareJSONEncoder,
                                  **json_opts)
            result += re.sub(r'"(\w+)": ', r'\1 = ', json_str[1:-1])
        return result

    @staticmethod
    def read(filename):
        """Read in a json datastructure and return a new AnalyticalPulse"""
        with open(filename, 'r') as in_fh:
            kwargs = json.load(in_fh, cls=NumpyAwareJSONDecoder)
            pulse = AnalyticalPulse(**kwargs)
        return pulse

    def pulse(self, tgrid=None, time_unit=None, ampl_unit=None, freq_unit=None,
              mode=None):
        """Return a QDYN.pulse.Pulse instance that contains the corresponding
        analytical pulse"""
        self._check_parameters()
        if tgrid is None:
            tgrid = pulse_tgrid(self.T, self.nt, self.t0)
        if time_unit is None:
            time_unit = self.time_unit
        if ampl_unit is None:
            ampl_unit = self.ampl_unit
        if freq_unit is None:
            freq_unit = self.freq_unit
        if mode is None:
            mode = self.mode
        amplitude = self._formulas[self._formula](tgrid, **self.parameters)
        if (not isinstance(amplitude, np.ndarray)
        and amplitude.ndim != 1):
            raise TypeError(('Formula "%s" returned type %s instead of the '
                             'required 1-D numpy array')%(
                             self._formula, type(amplitude)))
        if mode is None:
            if np.isrealobj(amplitude):
                mode = 'real'
            else:
                mode = 'complex'
        else:
            if mode == 'real' and not np.isrealobj(amplitude):
                if np.max(np.abs(amplitude.imag)) > 0.0:
                    raise ValueError("mode is 'real', but amplitude has "
                                     "non-zero imaginary part")

        pulse = Pulse(tgrid=tgrid, amplitude=amplitude, time_unit=time_unit,
                      ampl_unit=ampl_unit, freq_unit=freq_unit, mode=mode)
        pulse.preamble = [self.header, ]
        return pulse


def CRAB_carrier(t, time_unit, freq, freq_unit, a, b, normalize=False,
    complex=False):
    r'''
    Construct a "carrier" based on the CRAB formula

        .. math::
        E(t) = \sum_{n} (a_n \cos(\omega_n t) + b_n \cos(\omega_n t))

    where :math:`a_n` is the n'th element of `a`, :math:`b_n` is the n'th
    element of `b`, and :math:`\omega_n` is the n'th element of freq.

    Parameters
    ----------
    t : array-like
        time grid values
    time_unit : str
        Unit of `t`
    freq : scalar, ndarray(float64)
        Carrier frequency or frequencies
    freq_unit : str
        Unit of `freq`
    a: array-like
        Coefficients for cosines
    b: array-line
        Coefficients for sines
    normalize: logical, optional
        If True, normalize the resulting carrier such that its values are in
        [-1,1]
    complex: logical, optional
        If True, oscillate in the complex plane

        .. math::
        E(t) = \sum_{n} (a_n - i b_n) \exp(i \omega_n t)

    Notes
    -----

    `freq_unit` can be Hz (GHz, MHz, etc), describing the frequency directly,
    or any energy unit, in which case the energy value E (given through the
    freq parameter) is converted to an actual frequency as

     .. math:: f = E / (\\hbar * 2 * pi)
    '''
    from QDYN.units import NumericConverter
    convert = NumericConverter()
    c = convert.to_au(1, time_unit) * convert.to_au(1, freq_unit)
    assert len(a) == len(b) == len(freq), \
    "freq, a, b must all be of the same length"
    if complex:
        signal = np.zeros(len(t), dtype=np.complex128)
    else:
        signal = np.zeros(len(t), dtype=np.float64)
    for w_n, a_n, b_n in zip(freq, a, b):
        if complex:
            signal += (a_n -1j*b_n) * np.exp(1j*c*w_n*t)
        else:
            signal += a_n * np.cos(c*w_n*t) + b_n * np.sin(c*w_n*t)
    if normalize:
        signal *= 1.0 / (np.abs(signal).max())
    return signal


def ampl_field_free(tgrid):
    return 0.0 * carrier(tgrid, 'ns', 0.0, 'GHz').real


def ampl_1freq(tgrid, E0, T, w_L):
    return E0 * blackman(tgrid, 0, T) * carrier(tgrid, 'ns', w_L, 'GHz').real

def ampl_1freq_rwa(tgrid, E0, T, w_L, w_d):
    # note: amplitude reduction by 1/2 is included in construction of ham
    return E0 * blackman(tgrid, 0, T) \
           * carrier(tgrid, 'ns', (w_L-w_d), 'GHz', complex=True)


def ampl_1freq_0(tgrid, E0, T, w_L=0.0):
    return E0 * blackman(tgrid, 0, T) * carrier(tgrid, 'ns', w_L, 'GHz').real


def ampl_2freq(tgrid, E0, T, freq_1, freq_2, a_1, a_2, phi):
    return E0 * blackman(tgrid, 0, T) \
           * carrier(tgrid, 'ns', freq=(freq_1, freq_2),
                     freq_unit='GHz', weights=(a_1, a_2),
                     phases=(0.0, phi)).real

def ampl_2freq_rwa(tgrid, E0, T, freq_1, freq_2, a_1, a_2, phi, w_d):
    # note: amplitude reduction by 1/2 is included in construction of ham
    return E0 * blackman(tgrid, 0, T) \
           * carrier(tgrid, 'ns', freq=(freq_1-w_d, freq_2-w_d),
                     freq_unit='GHz', weights=(a_1, a_2),
                     phases=(0.0, phi), complex=True)


def ampl_5freq(tgrid, E0, T, freq_low, a_low, b_low, freq_high, a_high,
    b_high):
    norm_carrier = CRAB_carrier(tgrid, 'ns', freq_high, 'GHz', a_high, b_high,
                                normalize=True)
    crab_shape = CRAB_carrier(tgrid, 'ns', freq_low, 'GHz', a_low, b_low,
                              normalize=True)
    return E0 * blackman(tgrid, 0, T) * crab_shape * norm_carrier


def ampl_5freq_rwa(tgrid, E0, T, freq_low, a_low, b_low, freq_high, a_high,
    b_high, w_d):
    norm_carrier = CRAB_carrier(tgrid, 'ns', freq_high-w_d, 'GHz', a_high,
                                b_high, normalize=True, complex=True)
    crab_shape = CRAB_carrier(tgrid, 'ns', freq_low, 'GHz', a_low, b_low,
                              normalize=True)
    # note: amplitude reduction by 1/2 is included in construction of ham
    return E0 * blackman(tgrid, 0, T) * crab_shape * norm_carrier


AnalyticalPulse.register_formula('field_free', ampl_field_free)
AnalyticalPulse.register_formula('1freq',      ampl_1freq)
AnalyticalPulse.register_formula('2freq',      ampl_2freq)
AnalyticalPulse.register_formula('5freq',      ampl_5freq)
AnalyticalPulse.register_formula('1freq_rwa',  ampl_1freq_rwa)
AnalyticalPulse.register_formula('2freq_rwa',  ampl_2freq_rwa)
AnalyticalPulse.register_formula('5freq_rwa',  ampl_5freq_rwa)



def test():
    """Run a test of all pulse shapes"""
    import filecmp

    try:
        AnalyticalPulse.register_formula('1freq_0', 'bla')
    except TypeError as e:
        print(e)
    else:
        raise AssertionError("should catch non-callable formula")
    AnalyticalPulse.register_formula('1freq_0', ampl_1freq_0)
    try:
        AnalyticalPulse('1freq_0', T=200, nt=(200*11*100),
                        parameters={'E0': 100},
                        time_unit='ns', ampl_unit='MHz')
    except ValueError as e:
        print(e)
    else:
        raise AssertionError("constructor should catch missing parameter")

    try:
        AnalyticalPulse('1freq_0', T=200, nt=(200*11*100),
                        parameters={'E0': 100, 'T':200, 'extra': 0},
                        time_unit='ns', ampl_unit='MHz')
    except ValueError as e:
        print(e)
    else:
        raise AssertionError("constructor should catch extra parameter")

    p1 = AnalyticalPulse('field_free', T=200, nt=(200*11*100),
            parameters={}, time_unit='ns', ampl_unit='MHz')
    print(p1.header)
    p1.write('p1.json', pretty=True)
    p1.pulse().write('p1.dat')
    p1_copy = AnalyticalPulse.read('p1.json')
    p1_copy.write('p1_copy.json', pretty=True)
    if filecmp.cmp("p1.json", "p1_copy.json"):
        print("p1.json and p1_copy.json match")
    else:
        print("p1.json and p1_copy.json DO NOT MATCH")
        return 1

    p2 = AnalyticalPulse('1freq', T=200, nt=(200*11*100),
            parameters={'E0': 100, 'T': 200, 'w_L': 6.5},
            time_unit='ns', ampl_unit='MHz')
    print(p2.header)
    p2.write('p2.json', pretty=True)
    p2.pulse().write('p2.dat')
    p2_copy = AnalyticalPulse.read('p2.json')
    p2_copy.write('p2_copy.json', pretty=True)
    if filecmp.cmp("p2.json", "p2_copy.json"):
        print("p2.json and p2_copy.json match")
    else:
        print("p2.json and p2_copy.json DO NOT MATCH")
        return 1

    p3 = AnalyticalPulse('2freq', T=200, nt=(200*11*100),
            parameters={'E0': 100, 'T': 200, 'freq_1': 6.0, 'freq_2': 6.5,
                        'a_1': 0.5, 'a_2':1.0, 'phi': 0.0},
            time_unit='ns', ampl_unit='MHz')
    print(p3.header)
    p3.write('p3.json', pretty=True)
    p3.pulse().write('p3.dat')
    p3_copy = AnalyticalPulse.read('p3.json')
    p3_copy.write('p3_copy.json', pretty=True)
    if filecmp.cmp("p3.json", "p3_copy.json"):
        print("p3.json and p3_copy.json match")
    else:
        print("p3.json and p3_copy.json DO NOT MATCH")
        return 1

    freq = np.array([0.01, 0.0243, 8.32, 10.1, 5.3])
    a    = np.array([ 1.0, 0.21  , 0.58, 0.89, 0.1])
    b    = np.array([ 1.0, 0.51  , 0.09, 0.12, 0.71])
    p4 = AnalyticalPulse('5freq', T=200, nt=(200*11*100),
            parameters={'E0': 100, 'T': 200, 'freq': freq, 'a': a, 'b': b},
            time_unit='ns', ampl_unit='MHz')
    print(p4.header)
    p4.write('p4.json', pretty=True)
    p4.pulse().write('p4.dat')
    p4_copy = AnalyticalPulse.read('p4.json')
    assert isinstance(p4_copy.parameters['a'], np.ndarray), \
    "Coefficients 'a' should be a numpy array"
    p4_copy.write('p4_copy.json', pretty=True)
    if filecmp.cmp("p4.json", "p4_copy.json"):
        print("p4.json and p4_copy.json match")
    else:
        print("p4.json and p4_copy.json DO NOT MATCH")
        return 1


def main(argv=None):
    if argv is None:
        argv = sys.argv
    return test()

if __name__ == "__main__":
    sys.exit(main())
