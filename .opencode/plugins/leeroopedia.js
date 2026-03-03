/**
 * Leeroopedia plugin for OpenCode.ai
 *
 * Injects Leeroopedia bootstrap context via system prompt transform.
 * Skills are discovered via OpenCode's native skill tool from symlinked directory.
 */

import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Simple frontmatter extraction (avoid external dependencies for bootstrap)
const extractAndStripFrontmatter = (content) => {
  const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) return { frontmatter: {}, content };

  const frontmatterStr = match[1];
  const body = match[2];
  const frontmatter = {};

  for (const line of frontmatterStr.split('\n')) {
    const colonIdx = line.indexOf(':');
    if (colonIdx > 0) {
      const key = line.slice(0, colonIdx).trim();
      const value = line.slice(colonIdx + 1).trim().replace(/^["']|["']$/g, '');
      frontmatter[key] = value;
    }
  }

  return { frontmatter, content: body };
};

export const LeeroopediaPlugin = async ({ client, directory }) => {
  const leeroopediaSkillsDir = path.resolve(__dirname, '../../skills');

  // Helper to generate bootstrap content
  const getBootstrapContent = () => {
    const skillPath = path.join(leeroopediaSkillsDir, 'using-leeroopedia', 'SKILL.md');
    if (!fs.existsSync(skillPath)) return null;

    const fullContent = fs.readFileSync(skillPath, 'utf8');
    const { content } = extractAndStripFrontmatter(fullContent);

    const toolMapping = `**Tool Mapping for OpenCode:**
When skills reference tools you don't have, substitute OpenCode equivalents:
- \`Skill\` tool → OpenCode's native \`skill\` tool
- \`Read\`, \`Write\`, \`Edit\`, \`Bash\` → Your native tools

**Skills location:**
Leeroopedia skills are available via OpenCode's native \`skill\` tool.`;

    return `<EXTREMELY_IMPORTANT>
You have access to Leeroopedia.

**IMPORTANT: The using-leeroopedia skill content is included below. It is ALREADY LOADED - you are currently following it. Do NOT use the skill tool to load "using-leeroopedia" again - that would be redundant.**

${content}

${toolMapping}
</EXTREMELY_IMPORTANT>`;
  };

  return {
    // Use system prompt transform to inject bootstrap
    'experimental.chat.system.transform': async (_input, output) => {
      const bootstrap = getBootstrapContent();
      if (bootstrap) {
        (output.system ||= []).push(bootstrap);
      }
    }
  };
};
