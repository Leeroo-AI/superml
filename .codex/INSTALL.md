# SuperML Plugin — Codex Installation

## Prerequisites

### API Key (optional, recommended)

The plugin works without an API key — skills use web search to ground answers. With a key, your agent gets access to the Leeroopedia knowledge base (27k+ pages, faster and more precise lookups). The plugin will tell you if it's running without a key.

To get a key: [app.leeroopedia.com](https://app.leeroopedia.com/dashboard) — $20 free credit on signup, no credit card.

```bash
export LEEROOPEDIA_API_KEY=kpsk_your_key_here
```

Add to your shell profile (`~/.bashrc`, `~/.zshrc`) so it persists.

## Installation

1. Clone the plugin:
   ```bash
   git clone https://github.com/leeroo-ai/superml.git
   ```

2. Symlink skills into Codex's skill directory:
   ```bash
   mkdir -p ~/.agents/skills
   ln -sf "$(pwd)/superml/skills" ~/.agents/skills/superml
   ```

3. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package runner), then configure the MCP server. Add to `~/.codex/config.toml`:
   ```toml
   [mcp_servers.leeroopedia]
   command = "uvx"
   args = ["leeroopedia-mcp"]

   [mcp_servers.leeroopedia.env]
   LEEROOPEDIA_API_KEY = "kpsk_your_key_here"
   ```

4. Restart Codex.

## Verification

Ask Codex: "Search Leeroopedia for vLLM tensor parallelism configuration"

It should call the `search_knowledge` tool and return a grounded answer with `[PageID]` citations.
