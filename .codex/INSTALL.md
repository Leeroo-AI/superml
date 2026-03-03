# Leeroopedia Plugin — Codex Installation

## Prerequisites

1. Get your API key at [app.leeroopedia.com](https://app.leeroopedia.com)
2. Set the environment variable:
   ```bash
   export LEEROOPEDIA_API_KEY=kpsk_your_key_here
   ```

## Installation

1. Clone the plugin:
   ```bash
   git clone https://github.com/leeroo-ai/leeroopedia-plugin.git
   ```

2. Symlink skills into Codex's skill directory:
   ```bash
   mkdir -p ~/.agents/skills
   ln -sf "$(pwd)/leeroopedia-plugin/skills" ~/.agents/skills/leeroopedia
   ```

3. Configure the MCP server. Add to your Codex MCP config:
   ```json
   {
     "mcpServers": {
       "leeroopedia": {
         "command": "uvx",
         "args": ["leeroopedia-mcp"],
         "env": { "LEEROOPEDIA_API_KEY": "${LEEROOPEDIA_API_KEY}" }
       }
     }
   }
   ```

   Or use the remote URL (no Python required):
   ```bash
   # Replace YOUR_KEY with your kpsk_... key
   codex mcp add --transport http leeroopedia "https://mcp.leeroopedia.com/mcp?token=YOUR_KEY"
   ```

4. Restart Codex.

## Verification

Ask Codex: "Search Leeroopedia for vLLM tensor parallelism configuration"

It should call the `search_knowledge` tool and return a grounded answer with `[PageID]` citations.
