from inspect import getfullargspec
from typing import Callable, Dict, Set
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt


class GraphMaker:

    l_func = namedtuple("l_func", "f params")

    _n_unnamed_lambdas: int

    name: str
    xlabel: str
    ylabel: str

    _xlims: tuple
    _ylims: tuple

    _params: Set[str]

    _linspace_param: str
    _lines: Dict[str, Callable]

    def __init__(self,
                 name=None,
                 xlims=None,
                 ylims=None,
                 linspace_param="x"):

        self._n_unnamed_lambdas = 0

        self.name = name

        self.xlims = xlims
        self.ylims = ylims

        self._linspace_param = linspace_param

    @property
    def params(self):
        return frozenset(self._params)

    @property
    def xlims(self):
        return self._xlims

    @xlims.setter
    def xlims(self, lims):
        try:
            if isinstance(lims, tuple):
                self._xlims = (float(lims[0]) if lims[0] is not None
                               else -np.inf,
                               float(lims[1]) if lims[1] is not None
                               else np.inf)
            else:
                self._xlims = (-np.inf, float(lims))
        except TypeError:
            raise TypeError("limits must either be a tuple of elements or \
                a value that can be cast as a float")

    @property
    def ylims(self):
        return self._ylims

    @ylims.setter
    def ylims(self, lims):
        try:
            if isinstance(lims, tuple):
                self._ylims = (float(lims[0]) if lims[0] is not None
                               else -np.inf,
                               float(lims[1]) if lims[1] is not None
                               else np.inf)
            else:
                self._ylims = (-np.inf, float(lims))
        except TypeError:
            raise TypeError("limits must either be a tuple of elements or \
                a value that can be cast as a float")

    def set_xlabel(self, label):
        self.xlabel = str(label)
        return self

    def set_ylabel(self, label):
        self.ylabel = str(label)
        return self

    def define_line(self, _f, name=None):
        f_params = getfullargspec(_f).args
        self._params.update(f_params)

        if name is None:
            if _f.__name__ == "<lambda>":
                name = f"lambda_{self._n_unnamed_lambdas}"
                self._n_unnamed_lambdas += 1
            else:
                name = _f.__name__

        self._lines[name] = GraphMaker.l_func(f=_f, params=f_params)

        return self

    def plot_lines(self, ax, linspace, g_opts=None, plt_opts=None):

        linspace = linspace[(linspace > self.xlims[0])
                            &
                            (linspace < self.xlims[1])]

        g_opts = g_opts.copy()
        g_opts.update({self._linspace_param: linspace})

        plt_opts = plt_opts if plt_opts is not None else dict()

        for name, func in self._lines:

            call_params = {arg: val for arg, val in g_opts.items()
                           if arg in func.params.keys()}

            y_axis = func.f(**call_params)

            ax.plot(linspace, y_axis, label=name, **plt_opts)

        ax.legend()
        ax.set_title(self.name)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)


class EcoGraph:

    total_graphs = 0

    axis = namedtuple("axis", "maker")

    _linspace: np.ndarray

    _fig: plt.Figure
    _ax: Dict[str, namedtuple]

    _undefined: Set[str]
    _var_params: Dict[str, str]
    _static_params: Dict[str, str]

    def __init__(self, x_interval=None):

        EcoGraph.total_graphs += 1

        if len(x_interval) <= 3:
            self._linspace = np.arange(*x_interval)
        else:
            self._linspace = np.arange(x_interval)

        self._fig = None
        self._ax = None

        self._undefined = set()
        self._var_params = dict()
        self._static_params = dict()

    def __getattr__(self, attr):

        if attr in self._var_params:
            return self._var_params[attr]
        elif attr in self._static_params:
            return self._static_params[attr]
        else:
            raise AttributeError(f"attribute {attr} was not found in object \
                or parameters")

    @property
    def params(self):
        return (self.undefined_params
                .union(self.var_params)
                .union(self.static_params))

    @property
    def var_params(self):
        return frozenset(self._var_params.keys())

    @property
    def static_params(self):
        return frozenset(self._static_params.keys())

    @property
    def undefined_params(self):
        return frozenset(self._undefined)

    def set_static_vals(self, params):

        if not isinstance(params, dict):
            raise TypeError(f"argument params must be of type {dict} not \
                {type(params)}")

        for param, val in params:
            self.define_static_param(param, val)

        return self

    def set_var_defaults(self, defaults):

        if not isinstance(defaults, dict):
            raise TypeError(f"argument defaults must be of type {dict} not \
                {type(defaults)}")

        for param, val in defaults:
            self.define_var_param(param, val)

        return self

    def switch(self, param):

        if param in self.par_params:
            self._static_params[param] = self._var_params[param]
            del self._var_params[param]
        elif param in self.static_params:
            self._var_params[param] = self._static_params[param]
            del self._static_params[param]
        else:
            if param in self.undefined_params:
                raise ValueError(f"parameter {param} has not yet been \
                    defined")

            raise ValueError(f"parameter {param} is not part of any \
                defined graph")

    def define_var_param(self, param, value):

        if param in self.undefined_defaults:
            self._undefined.remove(param)
        elif param in self.static_defaults:
            del self._static_defaults[param]
        elif param not in self.var_defaults:
            raise ValueError(f"parameter {param} is not part of any \
                defined graph")

        self._var_defaults[param] = value

    def define_static_param(self, param, value):

        if param in self.undefined_params:
            self._undefined.remove(param)
        elif param in self.var_params:
            del self._var_params[param]
        elif param not in self.static_params:
            raise ValueError(f"parameter {param} is not part of any \
                defined graph")

        self._static_params[param] = value

    def set_figsize(self, figsize):
        w, h = figsize
        self._fig.set_size_inches(w, h=h)
        return self

    def define_graph(self, graph_maker, name=None):

        name = name if name is not None else f"Graph {EcoGraph.total_graphs}"

        self._fig, _ax = plt.subplots(1)
        self._ax = {name: EcoGraph.axis(axis=_ax, maker=graph_maker)}

    def __call__(self, **kwargs):
        pass


class MultiGraph(EcoGraph):

    total_graphs: int

    _ax_by_coord: np.ndarray  # of namedtuple

    def __init__(self, x_interval=None):
        # like old init but total_graphs is not class variable
        super()

    def change_name(self, old_name, new_name):
        pass

    def switch_coord(self, coord1, coord2):
        pass

    def get_maker_by_name(self, name):
        pass

    def get_maker_by_coord(self, coord):
        pass

    def get_ax_by_name(self, name):
        pass

    def get_ax_by_coord(self, coord):
        pass
