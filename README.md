[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/galacticglum/stimulator/main.svg)](https://results.pre-commit.ci/latest/github/galacticglum/stimulator/main)

# stimulator
Multimodal persona-aware conversational AI.

## Package manager
stimulator uses the [poetry](https://python-poetry.org/) package manager to manage its dependencies. To install the dependencies, run the following command:
```
poetry install
```
See the [poetry](https://python-poetry.org/) documentation for more information and
installation instructions.

#### Clone this repo and move into the directory
```shell
git clone https://github.com/galacticglum/stimulator.git
cd stimulator
```

#### Copy starter files
```shell
cp .env.example .env
```
The defaults are for running in *development* mode. Go through each variable in the file and make sure it is properly set. You will likely need to update the credentials. Once the file is updated, run
```shell
source .env
```
to load the environment variables into your shell.

If you'll be scraping data from Discord, you will need to set the `DISCORD_API_TOKEN` variable in the `.env` file with your personal access token. To acquire a token, you can either create a bot application in the [Discord Developer Portal](https://discord.com/developers/applications) and use its token, or you can use a self-token by following the instructions [here](https://gist.github.com/MarvNC/e601f3603df22f36ebd3102c501116c6). Note that using a self-token is not recommended and may violate Discord's terms of service, so proceed with caution.