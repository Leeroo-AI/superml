# SuperML Plugin — OpenCode Installation

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

3. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package runner), then configure MCP. Add to your `opencode.json`:
   ```json
   {
     "mcp": {
       "leeroopedia": {
         "type": "local",
         "command": ["uvx", "leeroopedia-mcp"],
         "environment": {
           "LEEROOPEDIA_API_KEY": "kpsk_your_key_here"
         }
       }
     }
   }
   ```

4. Restart OpenCode.

## Verification

Ask OpenCode: "Search Leeroopedia for QLoRA fine-tuning best practices"

It should call the `search_knowledge` tool and return a grounded answer with `[PageID]` citations.
