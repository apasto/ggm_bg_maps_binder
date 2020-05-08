#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute SH coefficients weights to perform 'tapering'
using a linear function or the 'gentle cut'
according proposed by Barthelmes, 2008
[http://icgem.gfz-potsdam.de/gentlecut_engl.pdf]
this weights can applied to SH coefficients shaped as:
(cosine/sine, l degree, m order), therefore (2, lmax+1, lmax+1)
by element-wise multiplication of coefficients and weights
"""

import numpy


def main():
    pass


def taper_weights(l_start, l_stop, l_max=None, taper='gentle'):
    """Compute SH tapering coefficients according to a linear or gentle-cut (Barthelmes, 2008) tapering.

    :param l_start: SH degree where tapering starts
    :param l_stop: SH degree where tapering stops
    :param l_max: maximum SH degree,
        defines the shape of returned array of weights, defaults to l_stop
    :param taper: type of computed taper,
        one of linear or gentle, defaults to gentle
    :type l_start: int
    :type l_stop: int
    :type l_max: int, optional
    :type taper: str, optional
    :return: an array of weights, shape is 'ci,l,m' : (2, l_max+1, l_max+1),
        which can be multiplied element-wise to SH coefficients
    :rtype: numpy.ndarray
    """
    if not(taper == 'linear' or taper == 'gentle'):
        raise AssertionError("taper must be either 'linear' or 'gentle', if provided")

    if l_max is None:
        l_max = l_stop
    elif l_max < l_stop:
        raise AssertionError(
            "l_stop (" + str(l_stop) + ") must be less than or equal to l_max (" + str(l_max) + ")")

    # first compute the weights l-degree wise, then broadcast to all m-orders and (c,i)
    # this implies that we populate weights even outside the SH coefficients 'triangle' - not an issue
    weights_lwise = numpy.ones(l_max + 1)

    if taper == 'linear':
        weights_lwise[l_start:l_stop+1] = numpy.linspace(1, 0, (l_stop - l_start + 1))
    elif taper == 'gentle':
        lg = numpy.linspace(l_start, l_stop, (l_stop - l_start + 1)) - l_start
        weights_lwise[l_start:l_stop+1] = \
            numpy.power(lg/(l_stop - l_start), 4) - 2*numpy.square(lg/(l_stop - l_start)) + 1

    if l_max > l_stop:
        weights_lwise[l_stop+1:] = numpy.zeros(l_max - (l_stop+1) + 1)

    # from l-degree wise vector to 'ci,l,m' shape : (2, l_max+1, l_max+1)
    # reshape is required to obtain l-wise tapering, not m-wise!
    weights = numpy.broadcast_to(
        numpy.reshape(weights_lwise, [weights_lwise.shape[0], 1]),
        [2, l_max+1, l_max+1])
    return weights


if __name__ == '__main__':
    main()
