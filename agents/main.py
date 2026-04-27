#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
from pathlib import Path
import re
import subprocess
from pydantic.dataclasses import dataclass
import yaml
from typing import Any

try:
    import readline

    # #143 UTF-8 backspace fix for macOS libedit
    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
    readline.parse_and_bind("set enable-meta-keybindings on")
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)


class SkillLoader:
    def __init__(self, skills_dir: Path) -> None:
        self.skills = {}
        self.skills_dir = skills_dir
        self._load_all()

    def _load_all(self):
        if not self.skills_dir.exists():
            print(f"Error: skills dir is not exists")
            return
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            try:
                text = f.read_text()
                meta, body = self._parse_frontmatter(text)
                name = meta.get("name", f.parent.name)
                self.skills[name] = {"meta": meta, "body": body}
            except Exception as e:
                print(f"Error: load {f.name} failed.", e)

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        match = re.search(
            r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL | re.MULTILINE
        )
        if not match:
            return {}, text
            # raise ValueError("Not a valid SKILL.md format")
        frontmatter_str, body = match.group(1), match.group(2)
        try:
            meta = yaml.safe_load(frontmatter_str) or {}
        except yaml.YAMLError:
            meta = {}
        return meta, body.strip()

    def get_description(self) -> str:
        if not self.skills:
            return "(no skills available)"
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f" - {name}: {desc} {f'[{tags}]' if tags else ''}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        skill = self.skills.get(name)
        if not skill:
            return f"Error: unknown skill '{name}'"
        return f'<skill name="{name}">\n{skill["body"]}\n</skill>'

class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list[dict[str, Any]]) -> str:
        if not items: 
            return "Error input schema"
        validated, in_progress_count = [], 0
        for item in items:
            status = item.get("status", "pending")
            if status == "in_progress":
                in_progress_count += 1
            if not hasattr(item, "id") or not hasattr(item, "text"):
                return "Error: item in items do not has correct attr"
            validated.append({"id": item["id"], "text": item["text"], "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "Nothing to do."

        marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
        return "\n".join(
            [f"{marker[item['status']]} {item['text']}" for item in self.items]
        )

if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("MODEL_ID"):
    raise ValueError("Plz set ANTHROPIC_AUTH_TOKEN or MODEL_ID first") 

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
    default_headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('ANTHROPIC_API_KEY')}",
    },
)
MODEL = os.environ["MODEL_ID"]
WORKDIR = Path.cwd()
SKILL_LOADER = SkillLoader(WORKDIR / "skills")

SYSTEM = f"""You are a coding agent at {os.getcwd()}. Make todolist before act, don't explain.
Skills available:
{SKILL_LOADER.get_description()}
"""
SUBAGENT_SYSTEM = f"You are a coding subagent created by father agent at {os.getcwd()}. Use bash to solve tasks. Make todolist before act, don't explain. After finished task, output the summy task result."

TODO = TodoManager()

CHILD_TOOLS = [
    {
        "name": "load_skill",
        "description": "load the skill content",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file's content.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "int"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content into file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Edit file's content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "todo",
        "description": "Update task list. Track progress on multi-step tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["id", "text", "status"],
                    },
                }
            },
            "required": ["items"],
        },
    },
]

PARENT_TOOLS = CHILD_TOOLS + [
    {
        "name": "task",
        "description": "Spawn a subagent with fresh context.",
        "input_schema": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
    }
]

TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo": lambda **kw: TODO.update(kw["items"]),
    "task": lambda **kw: run_subagent(kw["prompt"]),
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
}


def run_subagent(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    rounds_since_todo = 0

    # safe call
    for _ in range(30):
        response = client.messages.create(
            model=MODEL,
            system=SUBAGENT_SYSTEM,
            messages=messages,
            tools=CHILD_TOOLS,
            max_tokens=8000,
        )
        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        # If the model didn't call a tool, we're done
        if response.stop_reason != "tool_use":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "Finish! No summy!"
        # Execute each tool call, collect results
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33msubagent$ {block.name}\033[0m")
                handler = TOOL_HANDLERS.get(block.name)
                output = (
                    handler(**block.input) if handler else f"unknown tool {block.name}"
                )
                if block.name == "todo":
                    print(output)
                    rounds_since_todo = 0
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
        messages.append({"role": "user", "content": results})
        if rounds_since_todo >= 3 and messages:
            last = messages[-1]
            if last["role"] == "user" and isinstance(last.get("content"), list):
                last["content"].insert(
                    0,
                    {
                        "type": "text",
                        "text": "<reminder>Update your todos.</reminder>",
                    },
                )
        rounds_since_todo += 1

    messages.append(
        {
            "role": "user",
            "content": "just summize your current work progress and result and do nothing.",
        }
    )
    response = client.messages.create(
        model=MODEL,
        system=SUBAGENT_SYSTEM,
        messages=messages,
        tools=CHILD_TOOLS,
        max_tokens=8000,
    )
    return (
        "".join([block.text for block in response.content if hasattr(block, "text")])
        or "Not finished and no summy"
    )


def run_bash(command: str) -> str:
    if not command:
        return "Error: unexpected commmand"
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    #TODO: add more block
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def safe_path(p: str) -> Path:
    try:
        path = (Path.cwd() / p).resolve(strict=True)
    except Exception as e:
        raise e
    if not path.is_relative_to(Path.cwd()):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_read(path: str, limit: int | None = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    if not content:
        return f"Error: unexpected content"
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    rounds_since_todo = 0
    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=PARENT_TOOLS,
            max_tokens=8000,
        )
        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        # If the model didn't call a tool, we're done
        if response.stop_reason != "tool_use":
            return
        # Execute each tool call, collect results
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m$ {block.name}\033[0m")
                handler = TOOL_HANDLERS.get(block.name)
                output = (
                    handler(**block.input) if handler else f"unknown tool {block.name}"
                )
                if block.name == "todo":
                    print(output)
                    rounds_since_todo = 0
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
        messages.append({"role": "user", "content": results})
        if rounds_since_todo >= 3 and messages:
            last = messages[-1]
            if last["role"] == "user" and isinstance(last.get("content"), list):
                last["content"].insert(
                    0,
                    {
                        "type": "text",
                        "text": "<reminder>Update your todos.</reminder>",
                    },
                )
        rounds_since_todo += 1


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input(f"\033[36m{MODEL}>> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
