from nanomesh._doc import DocFormatterMeta, doc


def test_doc():

    @doc(number=456, field='rawr')
    def f():
        """This is a docstring {number} {field}."""
        pass

    @doc(f, number=123, field='moo')
    def g():
        pass

    assert f.__doc__ == 'This is a docstring 456 rawr.'
    assert g.__doc__ == 'This is a docstring 123 moo.'


def test_doc_formatter_meta():

    class Base(object, metaclass=DocFormatterMeta):

        @property
        def f():
            """{classname} property."""
            pass

        @classmethod
        def g():
            """{classname} classmethod."""
            pass

        def h():
            """{classname} method."""
            pass

    class Derived(Base):
        pass

    assert Derived.f.__doc__ == 'Derived property.'
    assert Derived.g.__doc__ == 'Derived classmethod.'
    assert Derived.h.__doc__ == 'Derived method.'

    assert Base.f.__doc__ == '{classname} property.'
    assert Base.g.__doc__ == '{classname} classmethod.'
    assert Base.h.__doc__ == '{classname} method.'
