import click
from experimaestro import parse_commandline

"""Defines the task command line argument prefix for experimaestro-handled command lines"""
TASK_PREFIX=["xpm", "--"]

@click.group()
def cli():
    """Main entry point for CLI"""
    pass

@cli.command(context_settings={"allow_extra_args": True})
@click.pass_context
def xpm(context):
    """Command used to run a task"""
    parse_commandline(context)


def forwardoption(argument, option_name):
    """Helper function
    
    Arguments:
        xpmtype {class or typename} -- The experimaestro type name or a class corresponding to a type
        option_name {str} -- The name of the option (or None if inferred from attribute)
    
    Raises:
        Exception -- Raised if the option `option_name` does not exist in the type, or if the type is not defined
    
    Returns:
        click.option -- Returns a click option annotation
    """

    xpmtype = argument.type
    name = "--%s" % (option_name or argument.name.replace("_", "-"))
    default = argument.defaultvalue
    # FIXME: type
    return click.option(name, help=argument.help, default=default)