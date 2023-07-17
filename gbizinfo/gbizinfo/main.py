import click

from gbizinfo.basic import name_and_address


def validate_corporate_number(ctx, _, corporate_number: str):
    if corporate_number and len(corporate_number) != 13:
        raise click.BadParameter("The length of corporate_number must be 13")
    return corporate_number


@click.command()
@click.option("--corporate_number", callback=validate_corporate_number, type=str)  # type: ignore
def main(corporate_number: str):
    print(name_and_address(corporate_number))


if __name__ == "__main__":
    main()
