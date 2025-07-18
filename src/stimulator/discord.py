"""Scrape chat messages from a Discord channel."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import discord
import discord.utils
import tqdm.asyncio as tqdm
import typer

from stimulator.utils import get_config_value, remove_urls

app = typer.Typer(help="Discord Scraper CLI")


class DiscordScraperClient(discord.Client):
    """Discord client for scraping messages from a channel."""

    # Private Instance Attributes
    #   _channel_id: The id of the channel to scrape.
    #   _limit: The maximum number of messages to scrape.
    #   _skip_bots: Whether to skip messages from bot users.
    #   _output_filepath: The filepath to save the scraped messages.
    _channel_id: int
    _limit: Optional[int]
    _skip_bots: bool
    _output_filepath: Optional[Path]

    def __init__(
        self,
        channel_id: int,
        limit: Optional[int] = None,
        skip_bots: bool = True,
        output_filepath: Optional[Path] = None,
    ) -> None:
        """Initialize the Discord scraper client."""
        super().__init__()
        self._channel_id = channel_id
        self._limit = limit
        self._skip_bots = skip_bots
        self._output_filepath = output_filepath

    async def on_ready(self) -> None:
        """Called when the client is ready."""
        discord.utils.setup_logging()

        channel = self.get_channel(self._channel_id)
        assert channel is not None, f"Channel with ID {self._channel_id} not found."
        assert isinstance(
            channel, discord.TextChannel
        ), "Channel must be a TextChannel."

        typer.echo(f"Scraping messages from channel: {channel.name} (ID: {channel.id})")
        history = channel.history(limit=self._limit, oldest_first=True)

        # Set the output filepath if not provided
        if self._output_filepath is None:
            self._output_filepath = Path(f"{channel.name}.discord.jsonl")

        typer.echo(f"Saving scraped messages to: {self._output_filepath}")
        with open(self._output_filepath, "w+") as fp:
            async for message in tqdm.tqdm(
                history,
                desc="Scraping messages",
                unit="message",
            ):
                if self._skip_bots and message.author.bot:
                    continue

                payload = {
                    "id": message.id,
                    "author": {
                        "id": message.author.id,
                        "username": message.author.name,
                        "display_name": message.author.display_name,
                        "bot": message.author.bot,
                    },
                    "content": message.content,
                    "clean_content": message.clean_content,
                    "created_at": datetime.timestamp(message.created_at),
                }
                json.dump(payload, fp)
                fp.write("\n")

        await self.close()


@app.command()
def scrape() -> None:
    """Scrape messages from a Discord channel and save them to a file."""
    # Get Discord user token
    token = get_config_value("DISCORD_USER_TOKEN", ask_user=True, secret=True)
    if not token:
        typer.echo("Error: Discord token is required.")
        raise typer.Exit(code=1)

    # Get target channel ID
    channel_id = get_config_value("DISCORD_CHANNEL_ID", ask_user=True)
    if not channel_id:
        typer.echo("Error: Discord channel ID is required.")
        raise typer.Exit(code=1)
    try:
        channel_id = int(channel_id)
    except ValueError:
        typer.echo("Error: Invalid Discord channel ID. It must be an integer.")
        raise typer.Exit(code=1)

    client = DiscordScraperClient(channel_id)
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(client.start(token))
    except Exception:
        pass


@app.command()
def to_pc_fmt(
    input_filepath: Path = typer.Argument(
        help="Path to the input JSONL file with scraped Discord messages.",
    ),
    output_filepath: Optional[Path] = typer.Option(
        None,
        help="Path to save the converted persona-chat format.",
    ),
    window_size: int = typer.Option(
        10, help="Number of messages to include in each conversation window."
    ),
    skip_bots: bool = typer.Option(
        True, help="Whether to skip messages from bot users."
    ),
    remove_links: bool = typer.Option(
        True, help="Whether to remove links from messages."
    ),
    escape_markdown: bool = typer.Option(
        True, help="Whether to escape markdown characters in messages."
    ),
) -> None:
    """Convert scraped Discord messages to persona-chat format."""
    if output_filepath is None:
        output_filepath = input_filepath.with_suffix(".pc.jsonl")
    typer.echo(f"Converting messages from {input_filepath} to {output_filepath}")

    messages = []
    with open(input_filepath) as fp:
        for line in fp:
            try:
                message = json.loads(line)
                if skip_bots and message["author"]["bot"]:
                    continue

                content = message["clean_content"] or message["content"]
                if remove_links:
                    content = remove_urls(content)
                # Remove leading and trailing whitespace
                content = content.strip()
                if not content:
                    continue
                if escape_markdown:
                    content = discord.utils.escape_markdown(content)

                messages.append(
                    {
                        "timestamp": message["created_at"],
                        "persona": message["author"]["username"],
                        "message": content,
                    }
                )
            except json.JSONDecodeError as e:
                typer.echo(f"Error decoding JSON: {e}")
                continue

    if not messages:
        typer.echo("No valid messages found to convert.")
        raise typer.Exit(code=1)
    typer.echo(f"Total messages to convert: {len(messages)}")

    with open(output_filepath, "w+") as fp:
        for i in tqdm.tqdm(range(window_size, len(messages))):
            history = messages[i - window_size: i]
            target = messages[i]
            example = {
                "history": [(msg["persona"], msg["message"]) for msg in history],
                "next_message": target["message"],
                "persona": target["persona"],
            }
            json.dump(example, fp)
            fp.write("\n")
