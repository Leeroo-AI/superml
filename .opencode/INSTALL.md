# SuperML Plugin — OpenCode Installation

## Prerequisites

1. Get your API key at [app.leeroopedia.com](https://app.leeroopedia.com)
2. Set the environment variable:
   ```bash
   export LEEROOPEDIA_API_KEY=kpsk_your_key_here
   ```

## Installation

1. Clone the plugin:
   ```bash
   git clone https://github.com/leeroo-ai/superml.git
   ```

2. Symlink the OpenCode plugin and skills:
   ```bash
   PLUGIN_DIR="$(pwd)/superml"
   OPENCODE_DIR="${OPENCODE_CONFIG_DIR:-$HOME/.config/opencode}"

   # Symlink plugin
   mkdir -p "$OPENCODE_DIR/plugins"
   ln -sf "$PLUGIN_DIR/.opencode/plugins/superml.js" "$OPENCODE_DIR/plugins/superml.js"

   # Symlink skills
   mkdir -p "$OPENCODE_DIR/skills"
   ln -sf "$PLUGIN_DIR/skills" "$OPENCODE_DIR/skills/superml"
   ```

3. Configure MCP. Add to your OpenCode config:
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

4. Restart OpenCode.

## Verification

Ask OpenCode: "Search Leeroopedia for QLoRA fine-tuning best practices"

It should call the `search_knowledge` tool and return a grounded answer with `[PageID]` citations.
