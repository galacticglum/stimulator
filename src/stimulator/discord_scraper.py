"""Scrape chat messages from a Discord channel."""

import os
from getpass import getpass

import typer

app = typer.Typer(help="Discord Scraper CLI")


def get_discord_token(env_var: str = "DISCORD_API_TOKEN") -> str:
    """Read the Discord token from the user.

    First, try to read from an environment variable. If not set, prompt
    the user to secretly enter the token.

    Args:
        env_var: The environment variable name to check for the token.

    Returns:
        The Discord token.
    """
    token = os.getenv(env_var)
    if not token:
        typer.echo(f"Environment variable '{env_var}' not set.")
        token = getpass("Please enter your Discord token: ")
    return token


@app.command()
def scrape(
    channel_id: str = typer.Argument(
        ..., help="The ID of the Discord channel to scrape"
    ),
    output_file: str = typer.Option(
        "output.json", help="File to save scraped messages"
    ),
):
    """Scrape messages from a Discord channel and save them to a file."""
    token = get_discord_token()
    if not token:
        typer.echo("Error: Discord token is required.")
        raise typer.Exit(code=1)
