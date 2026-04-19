# readthis

A command-line text-to-speech tool powered by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). Feed it plain text, a URL, piped input, or your clipboard, it extracts the content and reads it aloud. Audio generation and playback run on separate threads, so speech starts almost immediately rather than waiting for the full text to be synthesised.


## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv tool install git+https://github.com/realpacific/readthis
```

## Usage

```bash
# Plain text
readthis "Hello, this is a test"

# URL — extracts and reads the article
readthis https://example.com/article

# Piped / multiline input
echo "First line.\nSecond line." | readthis
cat article.txt | readthis

# No argument — reads from clipboard
readthis
```

### Options

```
readthis [input] [--voice VOICE] [--speed SPEED] [--lang LANG]
```

| Flag      | Default    | Description                            |
| --------- | ---------- | -------------------------------------- |
| `--voice` | `af_heart` | Voice name                             |
| `--speed` | `1.0`      | Speech speed                           |
| `--lang`  | `a`        | Language code (`a` = American English) |

### Examples

```bash
readthis "Good morning" --voice af_heart --speed 1.2
readthis https://example.com/blog-post --speed 1.5
readthis  # reads whatever is in your clipboard
```
