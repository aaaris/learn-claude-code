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

import json
import shlex
import platform
import os
from pathlib import Path
import re
import subprocess
from time import time
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
            return f"Error: unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
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
            if "id" not in item or "text" not in item:
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
KEEP_RECENT = 3
TRANSCRIPT_DIR = Path.cwd() / ".transcript"
THRESHOLD = 50000  # Token limit for triggering auto-compaction

SKILL_LOADER = SkillLoader(WORKDIR / "skills")

SYSTEM = f"""You are a coding agent at {os.getcwd()}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.

skills available:
{SKILL_LOADER.get_description()}
"""
SUBAGENT_SYSTEM = f"""You are a coding agent at {os.getcwd()}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.

skills available:
{SKILL_LOADER.get_description()}
"""

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
        "name": "compact",
        "description": "Manually compact conversation history, clean up expired tool results.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "task",
        "description": "Spawn a subagent with fresh context.",
        "input_schema": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
    },
]

PRESERVE_RESULT_TOOLS = {"read_file"}

TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo": lambda **kw: TODO.update(kw["items"]),
    "task": lambda **kw: run_subagent(kw["prompt"]),
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
    "compact": lambda **kw: "<compact>",
}


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    print(f"[tokens:{len(str(messages)) // 4}]")
    return len(str(messages)) // 4


def micro_compact(messages: list) -> list:
    tool_results = []
    tool_name_map = {}
    for i, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for j, part in enumerate(msg.get("content")):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((i, j, part))
        elif msg["role"] == "assistant" and isinstance(msg.get("content"), list):
            for block in msg.get("content"):
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_name_map[block.id] = block.name

    if len(tool_results) <= KEEP_RECENT:
        return messages

    for _, _, part in tool_results[:-KEEP_RECENT]:
        if not isinstance(part.get("content"), str) or len(part["content"]) <= 100:
            continue
        tool_id = part.get("tool_use_id", "")
        tool_name = tool_name_map.get(tool_id, "unknown")
        if tool_name in PRESERVE_RESULT_TOOLS:
            continue
        part["content"] = f"[Previous: used {tool_name}]"
    return messages


def auto_compact(messages: list) -> list:
    # Save transcript for recovery
    print("[auto_compact...]")
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")

    # LLM summarizes
    response = client.messages.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Summarize this conversation for continuity. Include:"
                + "1) What was accomplished, 2) Current state, 3) Key decisions made."
                + "Be concise but preserve critical details.\n\n"
                + json.dumps(messages, default=str)[-80000:],
            }
        ],
        max_tokens=2000,
    )
    summary = next(
        (block.text for block in response.content if hasattr(block, "text")), ""
    )
    if not summary:
        summary = "No summary generated."
    # Replace all messages with compressed summary
    return [
        {
            "role": "user",
            "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}",
        },
    ]


def run_subagent(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    rounds_since_todo = 0
    manual_todo = False

    # safe call
    for _ in range(30):
        micro_compact(messages)
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
        manual_compact = False
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compact":
                    manual_compact = True
                    print("Compressing...")
                    continue
                print(f"\033[33msubagent$ {block.name}\033[0m")
                handler = TOOL_HANDLERS.get(block.name)
                output = (
                    handler(**block.input) if handler else f"unknown tool {block.name}"
                )
                if block.name == "todo":
                    print(output)
                    manual_todo = True
                    rounds_since_todo = 0
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)
            break
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
        if manual_todo:
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
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# def run_bash(command: str) -> str:
#     if not command:
#         return "Error: unexpected commmand"
#     dangerous = ["rm", "sudo", "shutdown", "reboot", "del", "format", " attrib"]
#     try:
#         is_windows = platform.system() == "Windows"
#         args = shlex.split(command, posix=not is_windows)
#     except ValueError as e:
#         return f"Error: invalid command syntax: {e}"
#
#     if not args:
#         return "Error: empty command"
#
#     # Check for dangerous patterns in command name
#     cmd_name = args[0].lower()
#     for d in dangerous:
#         if d in cmd_name or cmd_name.endswith(d) or cmd_name == d:
#             return f"Error: Dangerous command blocked: {args[0]}"
#
#     # Check for redirects (Unix and Windows)
#     for arg in args[1:]:
#         if arg.startswith(">"):
#             return f"Error: Dangerous redirect blocked: {arg}"
#
#     try:
#         r = subprocess.run(
#             args,
#             shell=False,
#             cwd=os.getcwd(),
#             capture_output=True,
#             text=True,
#             timeout=120,
#         )
#         out = (r.stdout + r.stderr).strip()
#         return out[:50000] if out else "(no output)"
#     except subprocess.TimeoutExpired:
#         return "Error: Timeout (120s)"
#     except (FileNotFoundError, OSError) as e:
#         return f"Error: {e}"


def safe_path(p: str) -> Path:
    try:
        path = (Path.cwd() / p).resolve()
    except Exception as e:
        raise e
    # Verify path is within workspace
    if not path.is_relative_to(Path.cwd()):
        raise ValueError(f"Path escapes workspace: {p}")
    # Ensure resolved path is still within workspace
    if not path.resolve().is_relative_to(Path.cwd().resolve()):
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
    manual_todo = False
    while True:
        micro_compact(messages)
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
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
        manual_compact = False
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compact":
                    print("Compressing...")
                    manual_compact = True
                    continue
                print(f"\033[33m$ {block.name}\033[0m")
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = (
                        handler(**block.input)
                        if handler
                        else f"unknown tool {block.name}"
                    )
                except Exception as e:
                    output = f"Error: {e}"
                if block.name == "todo":
                    print(output)
                    manual_todo = True
                    rounds_since_todo = 0
                print(output[:200])
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    }
                )
        messages.append({"role": "user", "content": results})
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)
            return
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
        if manual_todo:
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
