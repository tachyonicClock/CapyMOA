import click
from jinja2 import Environment, FileSystemLoader
import capymoa  # Needed to initialize JPype with MOA #noqa: F401
from typing import Literal, Sequence
from com.github import javacliparser
import jpype
from dataclasses import dataclass
import re
from keyword import iskeyword
from pathlib import Path

from moa.options import (
    AbstractOptionHandler as JAbstractOptionHandler,
    ClassOption as JClassOption,
)

TYPE_MAPPINGS = {
    "moa.classifiers.core.driftdetection.ChangeDetector": (
        "MOADriftDetector",
        "cli_str_drift_detector",
    ),
    "moa.classifiers.core.splitcriteria.SplitCriterion": (
        "SplitCriterion",
        "_split_criterion_to_cli_str",
    ),
}

file_dir = Path(__file__).parent


def camel_to_snake(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


@dataclass
class Option:
    name: str
    """Variable name in snake case."""
    doc: str
    """Documentation string for the option."""
    cli_char: str
    """Command line interface character for the option."""
    type: str
    """Python type of the option."""
    option_type: (
        None
        | Literal[
            "IntOption",
            "FloatOption",
            "FlagOption",
            "MultiChoiceOption",
            "JClassOption",
        ]
    )
    """The type of the option in javacliparser."""
    default: str
    """Default value of the option as a string."""
    cli_func: str | None = None
    """The function to convert the option to a CLI string, if applicable."""

    @staticmethod
    def from_javacliparser(option: javacliparser.Option) -> "Option":
        """Convert a ``javacliparser.Option`` to an ``Option`` dataclass."""
        name = camel_to_snake(str(option.getName()))
        doc = str(option.getPurpose())
        cli_char = str(option.getCLIChar())
        type_ = "Any"
        default = "None"
        option_type = None
        cli_func = None

        if isinstance(option, javacliparser.IntOption):
            type_ = "int"
            default = f"{option.getValue()}"
            option_type = "IntOption"
        elif isinstance(option, javacliparser.FloatOption):
            type_ = "float"
            default = f"{option.getValue()}"
            option_type = "FloatOption"
        elif isinstance(option, javacliparser.FlagOption):
            type_ = "bool"
            default = "False"
            option_type = "FlagOption"
        elif isinstance(option, javacliparser.MultiChoiceOption):
            choices = ", ".join(f'"{choice}"' for choice in option.getOptionLabels())
            type_ = f"Literal[{choices}]"
            default = f'"{option.getChosenLabel()}"'
            definition_list = []
            for label, description in zip(
                option.getOptionLabels(), option.getOptionDescriptions()
            ):
                definition_list.append(f"* ``{label}``: {description}")
            doc += "\n\n" + "\n".join(definition_list)
            option_type = "MultiChoiceOption"
        elif isinstance(option, JClassOption):
            required_type = option.getRequiredType().getName()
            type_, cli_func = TYPE_MAPPINGS.get(
                required_type, (f"'{required_type}'", None)
            )
            default = f'"{option.getValueAsCLIString()}"'
            option_type = "JClassOption"
        else:
            raise NotImplementedError(f"Option type {type(option)} not implemented")

        # If the name is a Python keyword, append an underscore to avoid syntax errors.
        if iskeyword(name):
            name += "_"

        return Option(
            name=name,
            cli_char=cli_char,
            doc=doc,
            type=type_,
            default=default,
            option_type=option_type,
            cli_func=cli_func,
        )


def get_options(abstract_options: JAbstractOptionHandler) -> Sequence[Option]:
    """Get the options of an object as a list."""
    options = abstract_options.getOptions().getOptionArray()
    return [Option.from_javacliparser(opt) for opt in options]


@click.command()
@click.argument("java_learner", type=str, required=True)
@click.argument(
    "py_base_class", type=click.Choice(["MOAClassifier", "MOARegressor"]), required=True
)
def main(java_learner: str, py_base_class: str) -> None:
    environment = Environment(loader=FileSystemLoader(file_dir / "templates"))
    environment.filters["camel_to_snake"] = camel_to_snake

    # Construct the Java object
    j_object = jpype.JClass(java_learner)()

    # Render the template to stdout
    template = environment.get_template(f"{py_base_class}.py.jinja")
    print(
        template.render(
            options=get_options(j_object),
            j_class=java_learner.split(".")[-1],
            j_package=java_learner.rsplit(".", 1)[0],
            py_base_class=py_base_class,
        )
    )


if __name__ == "__main__":
    main()
