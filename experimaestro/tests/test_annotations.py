# Annotation specific tests

from typing import Optional, List
from experimaestro import config, argument, Argument
import experimaestro.core.types as types

# --- Test manual name for configuration

@config("annotations.b")
class B:
    pass

def test_fullname():
    assert str(B.__xpm__.identifier) == "annotations.b"


# --- Automatic name for configuration

@config()
class A:
    pass

def test_noname():
    assert str(A.__xpm__.identifier) == "experimaestro.tests.test_annotations.a"


# --- Use type annotations

def test_class_variable():
    @config("annotations.class_variable.config")
    class Config:
        x: Argument[int]
        y: Argument[float] = 2.3
        z: Argument[Optional[float]]
        t: Argument[List[float]]
        
    arg_x = Config.__xpm__.getArgument("x")
    assert arg_x.name == "x"
    assert isinstance(arg_x.type, types.IntType)
    assert arg_x.required

    arg_y = Config.__xpm__.getArgument("y")
    assert arg_y.name == "y"
    assert isinstance(arg_y.type, types.FloatType)
    assert arg_y.default == 2.3
    assert not arg_y.required

    arg_z = Config.__xpm__.getArgument("z")
    assert arg_z.name == "z"
    assert isinstance(arg_y.type, types.FloatType)
    assert not arg_z.required

    arg_t = Config.__xpm__.getArgument("t")
    assert arg_t.name == "t"
    assert isinstance(arg_t.type, types.ArrayType)
    assert isinstance(arg_t.type.type, types.FloatType)
    assert arg_t.required
