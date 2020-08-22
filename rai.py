import inspect

from typing import Callable, Dict, Iterator, Set
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display
from ipywidgets import fixed, interact, widgets


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
                 xlims=(-np.inf, np.inf),
                 ylims=(-np.inf, np.inf),
                 linspace_param="x"):

        self._n_unnamed_lambdas = 0

        self.name = name

        self.xlabel = None
        self.ylabel = None

        self.xlims = xlims
        self.ylims = ylims

        self._params = set()

        self._linspace_param = linspace_param
        self._lines = dict()

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
                self._xlims = (float(lims), np.inf)
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
                self._ylims = (float(lims), np.inf)
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
        f_params = inspect.getfullargspec(_f).args
        f_params.remove(self._linspace_param)

        self._params.update(f_params)

        if name is None:
            if _f.__name__ == "<lambda>":
                name = f"lambda_{self._n_unnamed_lambdas}"
                self._n_unnamed_lambdas += 1
            else:
                name = _f.__name__

        self._lines[name] = GraphMaker.l_func(f=_f, params=f_params)

        return self

    def plot_lines(self, ax, linspace, f_opts=None):

        linspace = linspace[(linspace > self.xlims[0])
                            &
                            (linspace < self.xlims[1])]

        f_opts = f_opts.copy()

        # note that this should behave well with ipywidget
        f_opts.update({self._linspace_param: linspace})

        for name, func in self._lines.items():

            call_params = {arg: val for arg, val in f_opts.items()
                           if arg in func.params}

            call_params.update({self._linspace_param: linspace})

            y_axis = func.f(**call_params)

            y_out_of_bounds = ((y_axis > self.ylims[0])
                               &
                               (y_axis < self.ylims[1]))

            linspace = linspace[y_out_of_bounds]
            y_axis = y_axis[y_out_of_bounds]

            ax.plot(linspace, y_axis, label=name)

        ax.legend()
        ax.set_title(self.name)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)


class EcoGraph:

    total_graphs = 0

    axis = namedtuple("axis", "axis maker")
    interactive_param = namedtuple("interactive_param", "orig widget")

    _linspace: np.ndarray

    _fig: plt.Figure
    _ax: Dict[str, namedtuple]  # of axis

    _undefined: Set[str]
    _var_params: Dict[str, namedtuple]  # of interactive_param
    _static_params: Dict[str, str]

    def __init__(self, x_interval=None):

        EcoGraph.total_graphs += 1

        if not isinstance(x_interval, tuple):
            raise TypeError(f"parameter x_interval must be of type {tuple}, \
                not {type(x_interval)}")

        self._linspace = np.arange(*x_interval)

        self._fig = None
        self._ax = None

        self._undefined = set()
        self._var_params = dict()
        self._static_params = dict()

    def __getattr__(self, attr):

        if attr in self._var_params:
            return self._var_params[attr].widget.value
        elif attr in self._static_params:
            return self._static_params[attr]
        else:
            raise AttributeError(f"attribute {attr} was not found in object \
                or parameters")

    @staticmethod
    def process_interactive(value, kind=None):

        INVALID = f"invalid interactive combination between a value of type \
            {type(value)} and a widget of kind {kind}"

        if isinstance(value, (int, float)):
            value = float(value)
            if kind == "text":
                return widgets.FloatText(
                    value=value,
                )
            elif kind == "slider":
                return widgets.FloatSlider(
                    value=value,
                    min=2*value*(value < 0),
                    max=2*value*(value > 0)
                )
            elif kind is not None:
                raise ValueError(INVALID)

        elif isinstance(value, tuple):

            # infer tuple range values
            if len(value) > 3:
                raise ValueError(f"tuple values should be at most of length \
                    3, not {len(value)}")
            elif len(value) == 1:
                return EcoGraph.process_interactive(value[0], kind=kind)

            elif len(value) >= 2:
                lower, default, upper = value[0], np.mean(value[:2]), value[1]

                step = value[2] if len(value) == 3 else 1

            # create interactive from infered values
            if kind == "text":
                return widgets.BoundedFloatText(
                    value=default,
                    min=lower,
                    max=upper,
                    step=step
                )
            elif kind == "slider":
                return widgets.FloatSlider(
                    value=default,
                    min=lower,
                    max=upper,
                    step=step
                )
            elif kind == "dropdown":
                return widgets.Dropdown(
                    options=range(int(lower), int(upper), int(step)),
                    value=int(default)
                )
            elif kind is not None:
                raise ValueError(INVALID)

        elif isinstance(value, list):

            if kind == "slider":
                return widgets.SelectionSlider(
                    options=value,
                    value=value[0]
                )
            elif kind == "dropdown":
                return widgets.Dropdown(
                    options=value,
                    value=value[0]
                )
            elif kind == "option":
                return widgets.RadioButtons(
                    options=value
                )
            elif kind is not None:
                raise ValueError(INVALID)

        elif isinstance(value, bool):
            return EcoGraph.process_interactive([value, not value])

        return value

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

    def define_var_param(self, param, value, kind=None):

        if param in self.undefined_params:
            self._undefined.remove(param)
        elif param in self.static_params:
            del self._static_defaults[param]
        elif param not in self.var_params:
            raise ValueError(f"parameter {param} is not part of any \
                defined graph")

        self._var_params[param] = EcoGraph.interactive_param(
            orig=value,
            widget=EcoGraph.process_interactive(value, kind=kind)
        )

        return self

    def define_static_param(self, param, value):

        if param in self.undefined_params:
            self._undefined.remove(param)
        elif param in self.var_params:
            del self._var_params[param]
        elif param not in self.static_params:
            raise ValueError(f"parameter {param} is not part of any \
                defined graph")

        self._static_params[param] = fixed(value)

        return self

    def set_static_vals(self, params):

        if not isinstance(params, dict):
            raise TypeError(f"argument params must be of type {dict} not \
                {type(params)}")

        for param, val in params.items():
            self.define_static_param(param, val)

        return self

    def make_static(self, param):

        if param not in self.var_params:
            raise ValueError(f"{param} is not currently a variable parameter")

        self.define_static_param(
            param,
            self._var_params[param].widget.value
        )

    def set_figsize(self, figsize):

        if not (isinstance(figsize, tuple) and len(figsize) == 2):
            raise ValueError("figsize must be a tuple of length two")

        _w, _h = figsize
        self._fig.set_size_inches(_w, h=_h)
        return self

    def define_graph(self, graph_maker, name=None):

        name = name if name is not None else f"Graph {EcoGraph.total_graphs}"

        # there can only be one graph in an EcoGraph object at once
        self._undefined = set(graph_maker.params)

        self._fig, _ax = plt.subplots(1)
        self._ax = {name: EcoGraph.axis(axis=_ax, maker=graph_maker)}

    def __call__(self, **kwargs):
        """
        """

        # this is a mechanic required by ipywidgets
        def plot_axes(**all_params):

            # name, ax = next(iter(self._ax.items()))
            for name, ax in self._ax.items():
                # clear
                ax.axis.clear()

                # plot
                ax.maker.plot_lines(
                    ax.axis,
                    self._linspace,
                    f_opts=all_params
                )
                ax.axis.set_title(name)

            # display
            display(self._fig)

        # notice that in the (unlikely) case of an overlap,
        # static_param takes prescedence over variable
        # and kwargs take precedence over all.

        all_params = {
            **{name: param.widget 
               for name, param in self._var_params.items()},
            **{name: param 
               for name, param in self._static_params.items()},
            **kwargs
        }

        # dict nesting is required to ensure no repeated kwargs
        interact(
            plot_axes,
            **all_params
        )


class MultiGraph(EcoGraph):

    total_graphs = 0

    _shape: tuple
    _avail_coord: Iterator[tuple]
    _total_subgraphs: int
    _ax_by_coord: np.ndarray  # of namedtuple

    title: str

    def __init__(self, shape, x_interval=None, title=None):

        # A MultiGraph still counts as a graph, so it should add to the
        # superclass counter
        super().__init__(x_interval=x_interval)

        # It also counts as it's own object, which should be counted as well
        MultiGraph.total_graphs += 1

        if not isinstance(shape, tuple):
            raise TypeError(f"argument shape must be of tupe {tuple}, \
                not {type(shape)}")

        self._shape = shape
        self._avail_coord = MultiGraph._make_avail_coord_generator(shape)

        # This counter is for the graphs within this instance
        self._total_subgraphs = 0

        self.title = title if title is not None \
            else f"MultiGraph {MultiGraph.total_graphs}"

        self._fig, axes = plt.subplots(*shape)
        self._ax_by_coord = np.array([
            EcoGraph.axis(axis=ax, maker=None) for ax in axes.reshape(-1)
        ]).reshape(shape)

    @staticmethod
    def _make_avail_coord_generator(shape):

        for i in range(shape[0]):
            for j in range(shape[1]):
                yield (i, j)

        raise RuntimeError("graph is full, either delete one graph \
            or redefine an existing one")

    @property
    def shape(self):
        return self._shape

    @property
    def total_subgraphs(self):
        return self._total_subgraphs

    def get_maker_by_name(self, name):
        return self._ax[name].maker

    def get_maker_by_coord(self, coord):
        return self._ax_by_coord[coord].maker

    def get_ax_by_name(self, name):
        return self._ax[name].axis

    def get_ax_by_coord(self, coord):
        return self._ax_by_coord[coord].axis

    def define_graph(self, graph_maker, name=None):

        if name in self._ax:
            raise NameError(f"name {name} is already taken, either \
                rename the existing graph or use a new name")

        ax = self._ax_by_coord[next(self.avail_coord)]
        ax.maker = graph_maker

        name = name if name is not None \
            else f"Subgraph {self.total_subgraphs}"

        self._ax[name] = ax

        self._undefined.update(set(graph_maker.params))

        return self

    def rename_graph(self, old_name, new_name):
        if old_name not in self._ax:
            raise NameError(f"name {old_name} is not a graph")

        self._ax[new_name] = self._ax[old_name]
        del self._ax[old_name]

    def switch_graph_coord(self, coord1, coord2):
        self._ax_by_coord[coord1], self._ax_by_coord[coord2] = \
            self._ax_by_coord[coord2], self._ax_by_coord[coord1]


if __name__ == "__main__":

    def utility(x, U, e1, e2):
        return (U/(x**e1))**(1/e2)

    def budget(x, I, p1, p2):
        return (I - x*p1)/p2 

    consumption_graph = GraphMaker(
        name="Model of Consumer Consumption",
        xlims=(0, 20),
        ylims=(0, 20),
    )

    consumption_graph.xlabel = "price/utility of good x"
    consumption_graph.ylabel = "price/utility of good y"

    consumption_graph.define_line(
        utility,
        name="Utility"
    )

    consumption_graph.define_line(
        budget,
        name="Budget"
    )

    consumption_model = EcoGraph(
        x_interval=(0, 15, 1)
    )

    consumption_model.define_graph(consumption_graph)

    consumption_model.set_static_vals({
        "e1": 1/2,
        "e2": 1/2,
        "p1": 1,
        "p2": 1,
    })

    consumption_model.define_var_param("U", 4, "text")
    consumption_model.define_var_param("I", 8, "slider")

    consumption_model.set_figsize((8, 12))

    consumption_model()
