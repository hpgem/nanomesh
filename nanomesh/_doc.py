from __future__ import annotations

import functools
import inspect
import string
from textwrap import dedent
from types import FunctionType, MethodType
from typing import Any, Callable, TypeVar


def copy_func(f: FunctionType) -> FunctionType:
    """Make a copy of function `f`."""
    g = FunctionType(f.__code__,
                     f.__globals__,
                     name=f.__name__,
                     argdefs=f.__defaults__,
                     closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


Formatter = string.Formatter()


class DocFormatterMeta(type):
    """Format docstrings with the classname of the derived class.

    Updates instances of `{classname}` to `classname`.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        return cls

    def __new__(mcls, classname, bases, cls_dict, **kwargs):
        cls = super().__new__(mcls, classname, bases, cls_dict, **kwargs)

        for name, method in inspect.getmembers(cls):
            is_private = name.startswith('__')
            is_none = (not method) or (not method.__doc__)

            if any((is_private, is_none)):
                continue

            is_format_string = len(list(Formatter.parse(method.__doc__))) > 1
            if not is_format_string:
                continue

            if inspect.isfunction(method) or inspect.ismethod(method):
                mcls.update_method_docstring(mcls,
                                             cls,
                                             method,
                                             name,
                                             classname=classname)
            elif isinstance(method, property):
                mcls.update_property_docstring(mcls,
                                               cls,
                                               method,
                                               name,
                                               classname=classname)
            else:
                pass

        return cls

    def update_method_docstring(mcls, cls, method, name, *, classname):
        """Copy method to subclass and replace tokens in parent method
        docstring."""
        bound_classname = method.__qualname__.split('.')[-2]
        if bound_classname == classname:
            return

        for parent in cls.mro()[1:]:
            if hasattr(parent, name):
                new_method = copy_func(method)
                new_method.__doc__ = method.__doc__.format(classname=classname)

                if inspect.ismethod(method):
                    new_method = MethodType(new_method, cls)

                setattr(cls, name, new_method)
                return

    def update_property_docstring(mcls, cls, method, name, *, classname):
        for f in method.fget, method.fset, method.fdel:
            if f:
                bound_classname = f.__qualname__.split('.')[-2]
                break

        if bound_classname == classname:
            return

        for parent in cls.mro():
            if hasattr(parent, name):
                doc = method.__doc__.format(classname=classname)
                new_property = property(fget=method.fget,
                                        fset=method.fset,
                                        fdel=method.fdel,
                                        doc=doc)
                setattr(cls, name, new_property)
                return


# `doc` is derived from pandas.util._decorators (1.4.1)
# https://github.com/pandas-dev/pandas/blob/main/LICENSE
FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)


def doc(*docstrings: str | Callable, **params) -> Callable[[F], F]:
    """
    A decorator take docstring templates, concatenate them and perform string
    substitution on it.
    This decorator will add a variable "_docstring_components" to the wrapped
    callable to keep track the original docstring template for potential usage.
    If it should be consider as a template, it will be saved as a string.
    Otherwise, it will be saved as callable, and later user __doc__ and dedent
    to get docstring.
    Parameters
    ----------
    *docstrings : str or callable
        The string / docstring / docstring template to be appended in order
        after default docstring under callable.
    **params
        The string which would be used to format docstring template.
    """

    def decorator(decorated: F) -> F:
        # collecting docstring and docstring templates
        docstring_components: list[str | Callable] = []
        if decorated.__doc__:
            docstring_components.append(dedent(decorated.__doc__))

        for docstring in docstrings:
            if hasattr(docstring, '_docstring_components'):
                # error: Item "str" of "Union[str, Callable[..., Any]]" has no
                # attribute "_docstring_components"
                # error: Item "function" of "Union[str, Callable[..., Any]]"
                # has no attribute "_docstring_components"
                docstring_components.extend(
                    docstring._docstring_components  # type: ignore[union-attr]
                )
            elif isinstance(docstring, str) or docstring.__doc__:
                docstring_components.append(docstring)

        # formatting templates and concatenating docstring
        decorated.__doc__ = ''.join([
            component.format(**params)
            if isinstance(component, str) else dedent(component.__doc__ or '')
            for component in docstring_components
        ])

        # error: "F" has no attribute "_docstring_components"
        decorated._docstring_components = (  # type: ignore[attr-defined]
            docstring_components)
        return decorated

    return decorator
