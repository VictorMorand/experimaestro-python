# Tests for identifier computation

import json
from pathlib import Path
from typing import Dict, List, Optional
from experimaestro import (
    config,
    Param,
    param,
    Config,
    Constant,
    Meta,
    Option,
    pathgenerator,
    Annotated,
    Task,
    LightweightTask,
)
from experimaestro.core.objects import (
    ConfigInformation,
    ConfigWalkContext,
    setmeta,
)
from experimaestro.scheduler.workspace import RunMode


class A(Config):
    a: Param[int]
    pass


class B(Config):
    a: Param[int]
    pass


class C(Config):
    a: Param[int] = 1
    b: Param[int]


class D(Config):
    a: Param[A]


class Float(Config):
    value: Param[float]


@config()
class Values:
    value1: Param[float]
    value2: Param[float]


def getidentifier(x):
    return x.__xpm__.identifier.all


def assert_equal(a, b, message=""):
    assert getidentifier(a) == getidentifier(b), message


def assert_notequal(a, b, message=""):
    assert getidentifier(a) != getidentifier(b), message


def test_int():
    assert_equal(A(a=1), A(a=1))


def test_different_type():
    assert_notequal(A(a=1), B(a=1))


def test_order():
    assert_equal(Values(value1=1, value2=2), Values(value2=2, value1=1))


def test_default():
    assert_equal(C(a=1, b=2), C(b=2))


def test_inner_eq():
    assert_equal(D(a=A(a=1)), D(a=A(a=1)))


def test_float():
    assert_equal(Float(value=1), Float(value=1))


def test_float2():
    assert_equal(Float(value=1.0), Float(value=1))


# --- Argument name


def test_name():
    """The identifier fully determines the hash code"""

    @config("test.identifier.argumentname")
    class Config0:
        a: Param[int]

    @config("test.identifier.argumentname")
    class Config1:
        b: Param[int]

    @config("test.identifier.argumentname")
    class Config3:
        a: Param[int]

    assert_notequal(Config0(a=2), Config1(b=2))
    assert_equal(Config0(a=2), Config3(a=2))


# --- Test option


def test_option():
    @config("test.identifier.option")
    class OptionConfig:
        a: Param[int]
        b: Option[int] = 1

    assert_notequal(OptionConfig(a=2), OptionConfig(a=1))
    assert_equal(OptionConfig(a=1, b=2), OptionConfig(a=1))
    assert_equal(OptionConfig(a=1, b=2), OptionConfig(a=1, b=2))


# --- Dictionnary


def test_identifier_dict():
    """Test identifiers of dictionary structures"""

    class B(Config):
        x: Param[int]

    class A(Config):
        bs: Param[Dict[str, B]]

    assert_equal(A(bs={"b1": B(x=1)}), A(bs={"b1": B(x=1)}))
    assert_equal(A(bs={"b1": B(x=1), "b2": B(x=2)}), A(bs={"b2": B(x=2), "b1": B(x=1)}))

    assert_notequal(A(bs={"b1": B(x=1)}), A(bs={"b1": B(x=2)}))
    assert_notequal(A(bs={"b1": B(x=1)}), A(bs={"b2": B(x=1)}))


# --- Ignore paths


@config()
class TypeWithPath:
    a: Param[int]
    path: Param[Path]


def test_path():
    """Path should be ignored"""
    assert_equal(TypeWithPath(a=1, path="/a/b"), TypeWithPath(a=1, path="/c/d"))
    assert_notequal(TypeWithPath(a=2, path="/a/b"), TypeWithPath(a=1, path="/c/d"))


# --- Test with added arguments


def test_pathoption():
    """Path arguments should be ignored"""

    @config("pathoption_test")
    class A_with_path:
        a: Param[int]
        path: Annotated[Path, pathgenerator("path")]

    @config("pathoption_test")
    class A_without_path:
        a: Param[int]

    assert_equal(A_with_path(a=1), A_without_path(a=1))


def test_identifier_enum():
    """Path arguments should be ignored"""
    from enum import Enum

    class EnumParam(Enum):
        FIRST = 0
        SECOND = 1

    class EnumConfig(Config):
        a: Param[EnumParam]

    assert_notequal(EnumConfig(a=EnumParam.FIRST), EnumConfig(a=EnumParam.SECOND))
    assert_equal(EnumConfig(a=EnumParam.FIRST), EnumConfig(a=EnumParam.FIRST))


def test_identifier_addnone():
    """Test the case of new parameter (with None default)"""

    class B(Config):
        x: Param[int]

    class A_with_b(Config):
        __xpmid__ = "defaultnone"
        b: Param[Optional[B]] = None

    class A(Config):
        __xpmid__ = "defaultnone"

    assert_equal(A_with_b(), A())
    assert_notequal(A_with_b(b=B(x=1)), A())


def test_defaultnew():
    """Path arguments should be ignored"""

    @param("b", type=int, default=1)
    @param(name="a", type=int)
    @config("defaultnew")
    class A_with_b:
        pass

    @param(name="a", type=int)
    @config("defaultnew")
    class A:
        pass

    assert_equal(A_with_b(a=1, b=1), A(a=1))
    assert_equal(A_with_b(a=1), A(a=1))


def test_taskconfigidentifier():
    """Test whether the embedded task arguments make the configuration different"""

    class MyConfig(Config):
        a: Param[int]

    class MyTask(Task):
        x: Param[int]

        def task_outputs(self, dep):
            return dep(MyConfig(a=1))

    assert_equal(
        MyTask(x=1).submit(run_mode=RunMode.DRY_RUN),
        MyTask(x=1).submit(run_mode=RunMode.DRY_RUN),
    )
    assert_notequal(
        MyTask(x=2).submit(run_mode=RunMode.DRY_RUN),
        MyTask(x=1).submit(run_mode=RunMode.DRY_RUN),
    )


def test_constant():
    """Test if constants are taken into account for signature computation"""

    @config("test.constant")
    class A1:
        version: Constant[int] = 1

    @config("test.constant")
    class A1bis:
        version: Constant[int] = 1

    assert_equal(A1(), A1bis())

    @config("test.constant")
    class A2:
        version: Constant[int] = 2

    assert_notequal(A1(), A2())


class MetaA(Config):
    x: Param[int]


def test_identifier_meta():
    """Test forced meta-parameter"""

    class B(Config):
        a: Param[MetaA]

    class C(Config):
        a: Meta[MetaA]

    class ArrayConfig(Config):
        array: Param[List[MetaA]]

    class DictConfig(Config):
        params: Param[Dict[str, MetaA]]

    # As meta
    assert_notequal(B(a=MetaA(x=1)), B(a=MetaA(x=2)))
    assert_equal(B(a=setmeta(MetaA(x=1), True)), B(a=setmeta(MetaA(x=2), True)))

    # As parameter
    assert_equal(C(a=MetaA(x=1)), C(a=MetaA(x=2)))
    assert_notequal(C(a=setmeta(MetaA(x=1), False)), C(a=setmeta(MetaA(x=2), False)))

    # Array with mixed
    assert_equal(
        ArrayConfig(array=[MetaA(x=1)]),
        ArrayConfig(array=[MetaA(x=1), setmeta(MetaA(x=2), True)]),
    )

    # Array with empty list
    assert_equal(ArrayConfig(array=[]), ArrayConfig(array=[setmeta(MetaA(x=2), True)]))

    # Dict with mixed
    assert_equal(
        DictConfig(params={"a": MetaA(x=1)}),
        DictConfig(params={"a": MetaA(x=1), "b": setmeta(MetaA(x=2), True)}),
    )


def test_identifier_meta_default_dict():
    class DictConfig(Config):
        params: Param[Dict[str, MetaA]] = {}

    assert_equal(
        DictConfig(params={}),
        DictConfig(params={"b": setmeta(MetaA(x=2), True)}),
    )

    # Dict with mixed
    assert_equal(
        DictConfig(params={"a": MetaA(x=1)}),
        DictConfig(params={"a": MetaA(x=1), "b": setmeta(MetaA(x=2), True)}),
    )


def test_identifier_meta_default_array():
    class ArrayConfigWithDefault(Config):
        array: Param[List[MetaA]] = []

    # Array (with default) with mixed
    assert_equal(
        ArrayConfigWithDefault(array=[MetaA(x=1)]),
        ArrayConfigWithDefault(array=[MetaA(x=1), setmeta(MetaA(x=2), True)]),
    )
    # Array (with default) with empty list
    assert_equal(
        ArrayConfigWithDefault(array=[]),
        ArrayConfigWithDefault(array=[setmeta(MetaA(x=2), True)]),
    )


def test_identifier_pre_task():
    class MyConfig(Config):
        pass

    class IdentifierPreLightTask(LightweightTask):
        pass

    class IdentifierPreTask(Task):
        x: Param[MyConfig]

    task = IdentifierPreTask(x=MyConfig()).submit(run_mode=RunMode.DRY_RUN)
    task_with_pre = (
        IdentifierPreTask(x=MyConfig())
        .add_pretasks(IdentifierPreLightTask())
        .submit(run_mode=RunMode.DRY_RUN)
    )
    task_with_pre_2 = (
        IdentifierPreTask(x=MyConfig())
        .add_pretasks(IdentifierPreLightTask())
        .submit(run_mode=RunMode.DRY_RUN)
    )
    task_with_pre_3 = IdentifierPreTask(
        x=MyConfig().add_pretasks(IdentifierPreLightTask())
    ).submit(run_mode=RunMode.DRY_RUN)

    assert_notequal(task, task_with_pre, "No pre-task")
    assert_equal(task_with_pre, task_with_pre_2, "Same parameters")
    assert_equal(task_with_pre, task_with_pre_3, "Pre-tasks are order-less")


def test_identifier_init_task():
    class MyConfig(Config):
        pass

    class IdentifierInitTask(LightweightTask):
        pass

    class IdentifierInitTask2(Task):
        pass

    class IdentierTask(Task):
        x: Param[MyConfig]

    task = IdentierTask(x=MyConfig()).submit(run_mode=RunMode.DRY_RUN)
    task_with_pre = IdentierTask(x=MyConfig()).submit(
        run_mode=RunMode.DRY_RUN,
        init_tasks=[IdentifierInitTask(), IdentifierInitTask2()],
    )
    task_with_pre_2 = IdentierTask(x=MyConfig()).submit(
        run_mode=RunMode.DRY_RUN,
        init_tasks=[IdentifierInitTask(), IdentifierInitTask2()],
    )
    task_with_pre_3 = IdentierTask(x=MyConfig()).submit(
        run_mode=RunMode.DRY_RUN,
        init_tasks=[IdentifierInitTask2(), IdentifierInitTask()],
    )

    assert_notequal(task, task_with_pre, "No pre-task")
    assert_equal(task_with_pre, task_with_pre_2, "Same parameters")
    assert_notequal(task_with_pre, task_with_pre_3, "Same parameters")


# --- Check configuration reloads


def check_reload(config):
    """Checks that the serialized configuration, when reloaded,
    gives the same identifier"""
    old_identifier = config.__xpm__.identifier.all

    # Get the data structure
    data = json.loads(config.__xpm__.__json__())

    # Reload the configuration
    new_config = ConfigInformation.fromParameters(
        data, as_instance=False, discard_id=True
    )
    assert new_config.__xpm__._full_identifier is None
    new_identifier = new_config.__xpm__.identifier.all

    assert new_identifier == old_identifier


class IdentifierReloadConfig(Config):
    id: Param[str]


def test_identifier_reload_config():
    # Creates the configuration
    check_reload(IdentifierReloadConfig(id="123"))


class IdentifierReload(Task):
    id: Param[str]

    def task_outputs(self, dep):
        return IdentifierReloadConfig(id=self.id)


class IdentifierReloadDerived(Config):
    task: Param[IdentifierReloadConfig]


def test_identifier_reload_taskoutput():
    """When using a task output, the identifier should not be different"""

    # Creates the configuration
    task = IdentifierReload(id="123").submit(run_mode=RunMode.DRY_RUN)
    config = IdentifierReloadDerived(task=task)
    check_reload(config)


class IdentifierReloadTask(Task):
    id: Param[str]


class IdentifierReloadTaskConfig(Config):
    x: Param[int]


class IdentifierReloadTaskDerived(Config):
    task: Param[IdentifierReloadTask]
    other: Param[IdentifierReloadTaskConfig]


def test_identifier_reload_task_direct():
    """When using a direct task output, the identifier should not be different"""

    # Creates the configuration
    task = IdentifierReloadTask(id="123").submit(run_mode=RunMode.DRY_RUN)
    config = IdentifierReloadTaskDerived(
        task=task, other=IdentifierReloadTaskConfig(x=2)
    )
    check_reload(config)


def test_identifier_reload_meta():
    """Test identifier don't change when using meta"""
    # Creates the configuration
    task = IdentifierReloadTask(id="123").submit(run_mode=RunMode.DRY_RUN)
    config = IdentifierReloadTaskDerived(
        task=task, other=setmeta(IdentifierReloadTaskConfig(x=2), True)
    )
    check_reload(config)


class LoopA(Config):
    param_b: Param["LoopB"]


class LoopB(Config):
    param_c: Param["LoopC"]


class LoopC(Config):
    param_a: Param["LoopA"]
    param_b: Param["LoopB"]


def test_identifier_loop():
    c = LoopC()
    b = LoopB(param_c=c)
    a = LoopA(param_b=b)
    c.param_a = a
    c.param_b = b

    configs = [a, b, c]
    identifiers = [[] for _ in configs]

    for i in range(len(configs)):
        context = ConfigWalkContext()
        configs[i].__xpm__.seal(context)
        assert all([c.__xpm__._sealed for c in configs])

        for j in range(len(configs)):
            identifiers[j].append(configs[j].__xpm__.identifier)

        configs[i].__xpm__.__unseal__()
        assert all([not c.__xpm__._sealed for c in configs])

    # Check
    for i in range(len(configs)):
        for j in range(1, len(configs)):
            assert identifiers[i][0] == identifiers[i][j]
