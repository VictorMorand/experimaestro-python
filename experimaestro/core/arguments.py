"""Management of the arguments (params, options, etc) associated with the XPM objects"""


class TypedArgument:
    """A type"""
    def __init__(self, type):
        self.type = type

class TypeHint: 
    def __getitem__(self, type):
        return TypedArgument(type)


class Argument:
    def __init__(
        self,
        name,
        type: "Type",
        required=None,
        help=None,
        generator=None,
        ignored=False,
        default=None,
        checker=None
    ):
        required = (default is None) if required is None else required
        if default is not None and required is not None and required:
            raise Exception(
                "argument '%s' is required but default value is given" % name
            )

        self.name = name
        self.help = help
        self.checker = checker
        self.type = type
        self.ignored = ignored
        if ignored is None:
            self.ignored = self.type.ignore
        else:
            self.ignored = False
        self.required = required
        self.default = default
        self.generator = generator

    def __repr__(self):
        return "argument[{name}:{type}]".format(**self.__dict__)


