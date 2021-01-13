"""Configuration and tasks"""

from pathlib import Path
import hashlib
import struct
import io
import fasteners
import os
import shutil
import inspect
import importlib
from typing import Dict, Set
from experimaestro.utils import logger
from experimaestro.constants import CACHEPATH_VARNAME
import sys
from contextlib import contextmanager


class Identifier:
    def __init__(self, all, main, sub):
        self.all = all
        self.main = main
        self.sub = sub


class HashComputer:
    """This class is in charge of computing a config/task identifier"""

    OBJECT_ID = b"\x00"
    INT_ID = b"\x01"
    FLOAT_ID = b"\x02"
    STR_ID = b"\x03"
    PATH_ID = b"\x04"
    NAME_ID = b"\x05"
    NONE_ID = b"\x06"
    LIST_ID = b"\x07"
    TASK_ID = b"\x08"

    def __init__(self):
        # Hasher for parameters
        self._hasher = hashlib.sha256()
        self._subhasher = None

    def identifier(self) -> Identifier:
        sub = None if self._subhasher is None else self._subhasher.digest()
        main = self._hasher.digest()
        if sub:
            h = hashlib.sha256()
            h.update(main)
            h.update(sub)
            all = h.digest()
        else:
            all = main

        return Identifier(all, main, sub)

    def _hashupdate(self, bytes, subparam):
        if subparam:
            # If subparam, creates a specific sub-hasher
            if self._subhasher is None:
                self._subhasher = hashlib.sha256()
            self._subhasher.update(bytes)
        else:
            self._hasher.update(bytes)

    def update(self, value, subparam=False):
        if value is None:
            self._hashupdate(HashComputer.NONE_ID, subparam=subparam)
        elif isinstance(value, float):
            self._hashupdate(HashComputer.FLOAT_ID, subparam=subparam)
            self._hashupdate(struct.pack("!d", value), subparam=subparam)
        elif isinstance(value, int):
            self._hashupdate(HashComputer.INT_ID, subparam=subparam)
            self._hashupdate(struct.pack("!q", value), subparam=subparam)
        elif isinstance(value, str):
            self._hashupdate(HashComputer.STR_ID, subparam=subparam)
            self._hashupdate(value.encode("utf-8"), subparam=subparam)
        elif isinstance(value, list):
            self._hashupdate(HashComputer.LIST_ID, subparam=subparam)
            self._hashupdate(struct.pack("!d", len(value)), subparam=subparam)
            for x in value:
                self.update(x, subparam=subparam)
        elif isinstance(value, Config):
            xpmtype = value.__xpmtype__
            self._hashupdate(HashComputer.OBJECT_ID, subparam=subparam)
            self._hashupdate(xpmtype.identifier.name.encode("utf-8"), subparam=subparam)

            # Add task parameters
            if value.__xpm__._task:
                self._hashupdate(HashComputer.TASK_ID, subparam=subparam)
                self.update(value.__xpm__._task, subparam=subparam)

            # Process arguments (sort by name to ensure uniqueness)
            arguments = sorted(xpmtype.arguments.values(), key=lambda a: a.name)
            for argument in arguments:
                arg_subparam = subparam or argument.subparam

                # Ignored argument
                if argument.ignored or argument.generator:
                    continue

                # Argument value
                argvalue = getattr(value, argument.name, None)
                if argument.default and argument.default == argvalue:
                    # No update if same value
                    continue

                # Hash name
                self.update(argument.name, subparam=arg_subparam)

                # Hash value
                self._hashupdate(HashComputer.NAME_ID, subparam=arg_subparam)
                self.update(argvalue, subparam=arg_subparam)

        else:
            raise NotImplementedError("Cannot compute hash of type %s" % type(value))


def updatedependencies(dependencies, value: "Config", path):
    """Search recursively jobs to add them as dependencies"""
    if isinstance(value, Config):
        if value.__xpmtype__.task:
            dependencies.add(value.__xpm__.dependency())
        else:
            value.__xpm__.updatedependencies(dependencies, path)
    elif isinstance(value, (list, set)):
        for el in value:
            updatedependencies(dependencies, el, path)
    elif isinstance(value, (str, int, float, Path)):
        pass
    else:
        raise NotImplementedError("update dependencies for type %s" % type(value))


class TaggedValue:
    def __init__(self, value):
        self.value = value


@contextmanager
def add_to_path(p):
    """Temporarily add a path to sys.path"""
    import sys

    old_path = sys.path
    sys.path = sys.path[:]
    sys.path.insert(0, p)
    try:
        yield
    finally:
        sys.path = old_path


class GenerationContext:
    """Context when generating values in configurations"""

    @property
    def path(self):
        """Returns the path of the job directory"""
        raise NotImplementedError()


class ConfigInformation:
    """Holds experimaestro information for a config (or task) instance"""

    # Set to true when loading from JSON
    LOADING = False

    def __init__(self, pyobject):
        # The underlying pyobject and XPM type
        self.pyobject = pyobject
        self.xpmtype = pyobject.__xpmtype__  # type: ObjectType

        # Meta-informations
        self._tags = {}
        self._initinfo = {}
        self._task = None  # type: Task

        # State information
        self.job = None
        self.dependencies = []
        self.setting = False

        # Cached information
        self._identifier = None
        self._validated = False
        self._sealed = False

    def set(self, k, v, bypass=False):
        if self._sealed:
            raise AttributeError("Object is read-only")

        try:
            argument = self.xpmtype.arguments.get(k, None)
            if argument:
                if not bypass and argument.generator:
                    raise AssertionError("Property %s is read-only" % (k))
                object.__setattr__(self.pyobject, k, argument.type.validate(v))
            else:
                raise AttributeError(
                    "Cannot set non existing attribute %s in %s" % (k, self.xpmtype)
                )
                # object.__setattr__(self, k, v)
        except:
            logger.error("Error while setting value %s" % k)
            raise

    def addtag(self, name, value):
        self._tags[name] = value

    def xpmvalues(self, generated=False):
        """Returns an iterarator over arguments and associated values"""
        for argument in self.xpmtype.arguments.values():
            if hasattr(self.pyobject, argument.name) or (
                generated and argument.generator
            ):
                yield argument, getattr(self.pyobject, argument.name, None)

    def tags(self, tags=None):
        if tags is None:
            tags = {}
        tags.update(self._tags)
        for argument, value in self.xpmvalues():
            if isinstance(value, Config):
                value.__xpm__.tags(tags)
        return tags

    def run(self):
        self.pyobject.execute()

    def init(self):
        self.pyobject._init()

    def validate(self):
        """Validate a value"""
        if not self._validated:
            self._validated = True

            # Check function
            if inspect.isfunction(self.xpmtype.originaltype):
                # Get arguments from XPM definition
                argnames = set()
                for argument, value in self.xpmvalues(True):
                    argnames.add(argument.name)

                # Get declared arguments from inspect
                spec = inspect.getfullargspec(self.xpmtype.originaltype)
                declaredargs = set(spec.args)

                # Arguments declared but not set
                notset = declaredargs.difference(argnames)
                notdeclared = argnames.difference(declaredargs)

                if notset or notdeclared:
                    raise ValueError(
                        "Some arguments were set but not declared (%s) or declared but not set (%s) at %s"
                        % (",".join(notdeclared), ",".join(notset), self._initinfo)
                    )

            # Check each argument
            for k, argument in self.xpmtype.arguments.items():
                if hasattr(self.pyobject, k):
                    value = getattr(self.pyobject, k)
                    if isinstance(value, Config):
                        value.__xpm__.validate()
                elif argument.required:
                    if not argument.generator:
                        raise ValueError(
                            "Value %s is required but missing when building %s at %s"
                            % (k, self.xpmtype, self._initinfo)
                        )

            # Use __validate__ method
            if hasattr(self.pyobject, "__validate__"):
                self.pyobject.__validate__()

    def seal(self, job: "experimaestro.scheduler.Job"):
        """Seal the object, generating values when needed, before scheduling the associated job(s)"""
        if self._sealed:
            return

        for k, argument in self.xpmtype.arguments.items():
            if argument.generator:
                self.set(k, argument.generator(job), bypass=True)
            elif hasattr(self.pyobject, k):
                v = getattr(self.pyobject, k)
                if isinstance(v, Config):
                    v.__xpm__.seal(job)
        self._sealed = True

    @property
    def identifier(self) -> Identifier:
        """Computes the unique identifier"""
        if self._identifier is None:
            hashcomputer = HashComputer()
            hashcomputer.update(self.pyobject)
            self._identifier = hashcomputer.identifier()
        return self._identifier

    def dependency(self):
        """Returns a dependency"""
        from experimaestro.scheduler import JobDependency

        assert self.job, f"{self.xpmtype} is a task but was not submitted"
        return JobDependency(self.job)

    def updatedependencies(
        self, dependencies: Set["experimaestro.dependencies.Dependency"], path
    ):
        for argument, value in self.xpmvalues():
            try:
                updatedependencies(dependencies, value, path + [argument.name])
            except:
                logger.error("While setting %s", path + [argument.name])
                raise

    def submit(self, workspace, launcher, dryrun=False):
        # --- Prepare the object
        if self.job:
            raise Exception("task %s was already submitted" % self)
        if not self.xpmtype.task:
            raise ValueError("%s is not a task" % self.xpmtype)

        self.validate()

        # --- Submit the job
        from experimaestro.scheduler import Job, Scheduler

        self.job = self.xpmtype.task(
            self.pyobject, launcher=launcher, workspace=workspace, dryrun=dryrun
        )
        self.seal(self.job)

        # --- Search for dependencies
        self.updatedependencies(self.job.dependencies, [])
        self.job.dependencies.update(self.dependencies)

        if not dryrun and Scheduler.CURRENT.submitjobs:
            other = Scheduler.CURRENT.submit(self.job)
            if other:
                # Just returns the other task
                return other.config
        else:
            logger.warning("Simulating: not submitting job %s", self.job)

        # Handle an output configuration
        if hasattr(self.pyobject, "config"):
            config = self.pyobject.config()
            config.__xpm__._task = self.pyobject
            return config

        return self.pyobject

    # --- Serialization

    @staticmethod
    def _outputjsonobjects(value, jsonout, context, serialized):
        """Output JSON objects"""
        # objects
        if isinstance(value, Config):
            value.__xpm__._outputjson_inner(jsonout, context, serialized)
        elif isinstance(value, list):
            for el in value:
                ConfigInformation._outputjsonobjects(el, jsonout, context, serialized)
        elif isinstance(value, dict):
            for name, el in value.items():
                ConfigInformation._outputjsonobjects(el, jsonout, context, serialized)
        elif isinstance(value, (Path, int, float, str)):
            pass
        else:
            raise NotImplementedError(
                "Cannot serialize objects of type %s", type(value)
            )

    @staticmethod
    def _outputjsonvalue(key, value, jsonout, context):
        """Output JSON value"""
        if isinstance(value, Config):
            with jsonout.subobject(*key) as obj:
                obj.write("type", "python")
                obj.write("value", id(value))
        elif isinstance(value, list):
            with jsonout.subarray(*key) as arrayout:
                for el in value:
                    ConfigInformation._outputjsonvalue([], el, arrayout, context)
        elif isinstance(value, dict):
            with jsonout.subobject(*key) as objectout:
                for name, el in value.items():
                    ConfigInformation._outputjsonvalue([name], el, objectout, context)
        elif isinstance(value, Path):
            with jsonout.subobject(*key) as objectout:
                objectout.write("type", "path")
                objectout.write("value", str(value))
        elif isinstance(value, (int, float, str)):
            jsonout.write(*key, value)
        else:
            raise NotImplementedError(
                "Cannot serialize objects of type %s", type(value)
            )

    def _outputjson_inner(self, jsonstream, context, serialized: Set[int]):
        # Skip if already serialized
        if id(self.pyobject) in serialized:
            return

        serialized.add(id(self.pyobject))

        # Serialize sub-objects
        for argument, value in self.xpmvalues():
            ConfigInformation._outputjsonobjects(value, jsonstream, context, serialized)

        with jsonstream.subobject() as objectout:

            # Serialize ourselves
            objectout.write("id", id(self.pyobject))
            if not self.xpmtype._package:
                objectout.write("file", str(self.xpmtype._file))

            objectout.write("module", self.xpmtype._module)
            objectout.write("type", self.xpmtype.originaltype.__name__)

            # Serialize identifier and typename
            # TODO: remove when not needed (cache issues)
            objectout.write("typename", self.xpmtype.name())
            objectout.write("identifier", self.identifier.all.hex())

            with objectout.subobject("fields") as jsonfields:
                for argument, value in self.xpmvalues():
                    ConfigInformation._outputjsonvalue(
                        [argument.name], value, jsonfields, value
                    )

    def outputjson(self, out: io.TextIOBase, context):
        """Outputs the json of this object

        The format is an array of objects
        {
            "objects": [
                {
                    "id": <ID of the object>,
                    "filename": <filename>, // if in a file
                    "module": <module>, // if in a module
                    "type": <type>, // the type within the module or file
                    "fields":
                        { "key":  {"type": <type>, "value": <value>} }
                }
            ]

        }

        <type> is either a base type or a "python"

        The last object is the one that is serialized

        Arguments:
            out {io.TextIOBase} -- The output stream
            context {[type]} -- the command context
        """
        import jsonstreams

        serialized: Set[int] = set()
        with jsonstreams.Stream(jsonstreams.Type.object, fd=out, close_fd=True) as out:
            # Write information
            out.write("has_subparam", self.identifier.sub is not None)

            with out.subobject("tags") as objectout:
                for key, value in self.tags().items():
                    objectout.write(key, value)

            # Write objects
            with out.subarray("objects") as arrayout:
                self._outputjson_inner(arrayout, context, serialized)

    # def _outputjson(self, jsonout, context, key=[]):
    #     with jsonout.subobject(*key) as objectout:
    #         self._outputjson_inner(objectout, context)

    @staticmethod
    def _objectFromParameters(value, objects):
        if isinstance(value, list):
            return [ConfigInformation._objectFromParameters(x, objects) for x in value]
        if isinstance(value, dict):
            if value["type"] == "python":
                return objects[value["value"]]
            if value["type"] == "path":
                return Path(value["value"])
            else:
                raise Exception("Unhandled type: %s", value["type"])

        return value

    @staticmethod
    def fromParameters(definitions):
        o = None
        objects = {}

        for definition in definitions:
            module_name = definition["module"]

            # Avoids problem when runing module
            if module_name == "__main__":
                module_name = "_main_"

            if "file" in definition:
                path = definition["file"]
                with add_to_path(str(Path(path).parent)):
                    spec = importlib.util.spec_from_file_location(module_name, path)
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = mod
                    spec.loader.exec_module(mod)
            else:
                logger.debug("Importing module %s", definition["module"])
                mod = importlib.import_module(module_name)

            cls = getattr(mod, definition["type"])
            o = cls()

            if "typename" in definition:
                o.__xpmtypename__ = definition["typename"]
                o.__xpmidentifier__ = definition["identifier"]

            istaskdef = isinstance(o, BaseTaskFunction)
            if istaskdef:
                o.__arguments__ = set()

            for name, value in definition["fields"].items():
                v = ConfigInformation._objectFromParameters(value, objects)
                if istaskdef:
                    o.__arguments__.add(name)
                setattr(o, name, v)

            postinit = getattr(o, "__postinit__", None)
            if postinit is not None:
                postinit()

            assert definition["id"] not in objects, "Duplicate id %s" % definition["id"]
            objects[definition["id"]] = o

        return o

    @staticmethod
    def _fromPython(context: GenerationContext, param, value, objects):
        if hasattr(param, "generator") and param.generator:
            return param.generator(context)

        if isinstance(value, Config):
            return value.__xpm__._fromConfig(context, objects)

        if isinstance(value, list):
            return [
                ConfigInformation._fromPython(context, param.type.type, x, objects)
                for x in value
            ]

        return value

    def _fromConfig(self, context: GenerationContext, objects: Dict[int, object]):
        o = objects.get(id(self), None)
        if o is not None:
            return o

        o = object.__new__(self.xpmtype.originaltype)
        objects[id(self)] = o

        for key, param in self.xpmtype.arguments.items():
            value = self._fromPython(
                context, param, getattr(self.pyobject, key, None), objects
            )
            setattr(o, key, value)
            postinit = getattr(o, "__postinit__", None)
            if postinit is not None:
                postinit()

        return o

    def fromConfig(self, context: GenerationContext):
        """Generate an instance given the current configuration"""
        self.validate()

        return self._fromConfig(context, {})

    def add_dependencies(self, *dependencies):
        self.dependencies.extend(dependencies)


def clone(v):
    """Clone a value"""
    if isinstance(v, (str, float, int)):
        return v
    if isinstance(v, list):
        return [clone(x) for x in v]

    if isinstance(v, Config):
        # Create a new instance
        kwargs = {
            argument.name: clone(value) for argument, value in v.__xpm__.xpmvalues()
        }

        config = type(v).__new__(type(v))
        return type(v)(**kwargs)

    raise NotImplementedError("For type %s" % v)


def cache(fn, name: str):
    def __call__(config, *args, **kwargs):
        # Get path and create directory if needed
        hexid = config.__xpmidentifier__  # type: str
        typename = config.__xpmtypename__  # type: str
        dir = Path(os.environ[CACHEPATH_VARNAME]) / typename / hexid

        tmpdir = None
        if not dir.exists():
            tmpdir = dir.with_suffix(".tmp")
            if tmpdir.exists():
                logger.warning("Removing old temporary cache dir %s", tmpdir)
                shutil.rmtree(tmpdir)
            tmpdir.mkdir(parents=True, exist_ok=True)

        path = (tmpdir or dir) / name
        ipc_lock = fasteners.InterProcessLock(path.with_suffix(path.suffix + ".lock"))
        with ipc_lock:
            r = fn(config, path, *args, **kwargs)
            if tmpdir:
                # Renames to final directory since we succeeded
                tmpdir.rename(dir)
            return r

    return __call__


class Config:
    """Base type for all objects in python interface"""

    # Set to true when executing a task to remove all checks
    TASKMODE = False
    STACK_LEVEL = 1

    def __init__(self, **kwargs):
        # Add configuration
        from .types import ObjectType

        self.__xpmtype__ = self.__class__.__xpm__
        if not isinstance(self.__xpmtype__, ObjectType):
            logger.error("%s is not an object type", self.__xpmtype__)
            assert isinstance(self.__xpmtype__, ObjectType)

        xpm = ConfigInformation(self)
        caller = inspect.getframeinfo(inspect.stack()[self.STACK_LEVEL][0])
        xpm._initinfo = "%s:%s" % (str(Path(caller.filename).absolute()), caller.lineno)

        self.__xpm__ = xpm

        # Initialize with arguments
        for name, value in kwargs.items():
            # Check if argument is OK
            if name not in xpm.xpmtype.arguments:
                raise ValueError(
                    "%s is not an argument for %s" % (name, self.__xpmtype__)
                )

            # Special case of a tagged value
            if isinstance(value, TaggedValue):
                value = value.value
                self.__xpm__._tags[name] = value

            # Really set the value
            xpm.set(name, value, bypass=ConfigInformation.LOADING)

        # Initialize with default arguments
        for name, value in self.__xpmtype__.arguments.items():
            if name not in kwargs and value.default is not None:
                self.__xpm__.set(name, clone(value.default))

    def __setattr__(self, name, value):
        if not Config.TASKMODE and not name.startswith("__xpm"):
            return self.__xpm__.set(name, value)
        return super().__setattr__(name, value)

    def tag(self, name, value) -> "Config":
        self.__xpm__.addtag(name, value)
        return self

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        for argument, value in self.__xpm__.xpmvalues():
            if value != getattr(other, argument.name, None):
                return False
        return True

    def __arguments__(self):
        """Returns a map containing all the arguments"""
        return {arg.name: value for arg, value in self.__xpm__.xpmvalues()}

    def tags(self):
        """Returns the tag associated with this object (and below)"""
        return self.__xpm__.tags()

    def add_dependencies(self, *dependencies):
        """Adds tokens to the task"""
        self.__xpm__.add_dependencies(*dependencies)
        return self

    def instance(self, context: GenerationContext = None):
        """Return an instance with the current values"""
        return self.__xpm__.fromConfig(context)


class Task(Config):
    """Base type for all tasks"""

    def submit(self, *, workspace=None, launcher=None, dryrun=False):
        """Submit this task"""
        return self.__xpm__.submit(workspace, launcher, dryrun=dryrun)

    def stdout(self):
        return self.__xpm__.job.stdout

    def stderr(self):
        return self.__xpm__.job.stderr

    @property
    def job(self):
        return self.__xpm__.job

    @property
    def jobpath(self):
        if self.__xpm__.job:
            return self.__xpm__.job.jobpath
        raise AssertionError("Cannot ask the job path of a non submitted task")


class BaseTaskFunction(Task):
    STACK_LEVEL = 2

    """Useful to identify a task function"""

    def __init__(self, **kwargs):
        if not Config.TASKMODE:
            super().__init__(**kwargs)


# XPM task as a function
def gettaskclass(function, parents):
    class TaskFunction(*parents, BaseTaskFunction):
        def execute(self):
            kwargs = {a: getattr(self, a) for a in self.__arguments__}
            function(**kwargs)

    return TaskFunction
