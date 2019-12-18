# Import Python modules

import json
import sys
import inspect
import os.path as op
import os
import logging
import pathlib
from pathlib import Path, PosixPath
from typing import Union, Dict

import experimaestro.api as api
from .api import XPMObject, Typename
from .utils import logger
from .workspace import Workspace

# --- Annotations to define tasks and types
        
class config:

    """Annotations for experimaestro types"""
    def __init__(self, typename=None, description=None, parents=[]):
        """[summary]
        
        Keyword Arguments:
            typename {Typename, str} -- Unique identifier of the type (default: {None} will use the module/class name)
            description {str} -- Description of the config/task (default: {None})
            parents {list} -- Parent classes if annotating a method (default: {[]})
        """
        super().__init__()
        self.typename = typename
        self.description = description
        self.parents = parents

    def __call__(self, tp, basetype=api.XPMObject):
        # Check if conditions are fullfilled
        if self.typename is None:
            self.typename = Typename("%s.%s" % (tp.__module__.lower(), tp.__name__.lower()))

        originaltype = tp
        
        # --- If this is a method, encapsulate
        if inspect.isfunction(tp):
            tp = api.getfunctionpyobject(tp, self.parents, basetype=basetype)
        
        # --- Add XPMObject as an ancestor of t if needed
        elif inspect.isclass(tp):
            assert not self.parents, "parents can be used only for functions"
            if not issubclass(tp, basetype):
                __bases__ = (basetype, )
                if tp.__bases__ != (object, ):
                    __bases__ += tp.__bases__
                __dict__ = {key: value for key, value in tp.__dict__.items() if key not in ["__dict__"]}
                tp = type(tp.__name__, __bases__, __dict__)


        else:
            raise ValueError("Cannot use type %s as a type/task" % tp)

        logging.debug("Registering %s", self.typename)
        
        objecttype = api.ObjectType.create(tp, self.typename, self.description)
        tp.__xpm__ = objecttype
        objecttype.originaltype = originaltype
        
        return tp


class Array(api.TypeProxy):
    """Array of object"""
    def __init__(self, type):
        self.type = api.Type.fromType(type)

    def __call__(self):
        return api.ArrayType(self.type)

class Choice(api.TypeProxy):
    """A string with a choice among several alternative"""
    def __init__(self, *args):
        self.choices = args

    def __call__(self):
        return api.StringType


class task(config):
    """Register a task"""
    def __init__(self, typename=None, scriptpath=None, pythonpath=None, description=None, associate=None):
        super().__init__(typename, description)
        self.pythonpath = sys.executable if pythonpath is None else pythonpath
        self.scriptpath = scriptpath

    def __call__(self, objecttype):
        import experimaestro.commandline as commandline

        # Register the type
        objecttype = super().__call__(objecttype, basetype=api.XPMTask) 

        # Construct command  
        _type = objecttype.__xpm__.originaltype
        command = commandline.Command()
        command.add(commandline.CommandPath(self.pythonpath))
        command.add(commandline.CommandString("-m"))
        command.add(commandline.CommandString("experimaestro"))
        command.add(commandline.CommandString("run"))

        if _type.__module__ and _type.__module__ != "__main__":
            logger.debug("task: using module %s [%s]", _type.__module__, _type)
            command.add(commandline.CommandString(_type.__module__))
        else:
            filepath = Path(inspect.getfile(_type)).absolute()
            logger.debug("task: using file %s [%s]", filepath, _type)
            command.add(commandline.CommandString("--file"))
            command.add(commandline.CommandPath(filepath))

        command.add(commandline.CommandString(str(self.typename)))
        command.add(commandline.CommandParameters())
        commandLine = commandline.CommandLine()
        commandLine.add(command)

        objecttype.__xpm__.task = commandline.CommandLineTask(commandLine)
        return objecttype


# --- argument related annotations

class argument():
    """Defines an argument for an experimaestro type"""
    def __init__(self, name, type=None, default=None, required=None,
                 ignored=None, help=None):
        # Determine if required
        self.name = name                
        self.type = api.Type.fromType(type) if type else None
        self.help = help
        self.ignored = ignored
        self.default = default
        self.required = required
        self.generator = None

    def __call__(self, tp):
        # Get type from default if needed
        if self.type is None:
            if self.default is not None: 
                self.type = api.Type.fromType(type(self.default))

        # Type = any if no type
        if self.type is None:
            self.type = api.Any

        argument = api.Argument(self.name, self.type, help=self.help, required=self.required, ignored=self.ignored, generator=self.generator, default=self.default)
        tp.__xpm__.addArgument(argument)
        return tp

class pathargument(argument):
    """Defines a an argument that will be a relative path (automatically
    set by experimaestro)"""
    def __init__(self, name, path, help=""):
        """
        :param name: The name of argument (in python)
        :param path: The relative path
        """
        super().__init__(name, type=Path, help=help)
        self.generator = lambda jobcontext: jobcontext.jobpath / path


class ConstantArgument(argument):
    """
    An constant argument (useful for versionning tasks)
    """
    def __init__(self, name: str, value, xpmtype=None, help=""):
        super().__init__(name, type=xpmtype or api.Type.fromType(type(value)), help=help)
        self.generator = lambda jobcontext: api.clone(value)

class TaggedValue:
    def __init__(self, name: str, value):
        self.name = name
        self.value = value

def tag(name: str, value):
    """Tag a value"""
    return TaggedValue(name, value)

def tags(value):
    """Return the tags associated with a value"""
    if isinstance(value, Value):
        return value.tags()
    return value.__xpm__.sv.tags()

def tagspath(value: api.XPMObject):
    """Return the tags associated with a value"""
    p = Path()
    for key, value in value.__xpm__.sv.tags().items():
        p /= "%s=%s" % (key.replace("/","-"), value)
    return p



# --- Deprecated

def deprecateClass(klass):
    import inspect
    def __init__(self, *args, **kwargs):
        frameinfo = inspect.stack()[1]
        logger.warning("Class %s is deprecated: use %s in %s:%s (%s)", 
            klass.__name__, klass.__bases__[0].__name__, 
            frameinfo.filename, frameinfo.lineno,
            frameinfo.code_context
        )
        super(klass, self).__init__(*args, **kwargs)
        
    klass.__init__ = __init__
    return klass
