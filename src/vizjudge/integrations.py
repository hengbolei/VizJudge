"""Install the bundled agent skill and print MCP client configuration for host agents."""

from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import sys
from importlib.resources import as_file, files
from pathlib import Path

from vizjudge._version import __version__

SKILL_NAME = "vizjudge"
MCP_SERVER_NAME = "vizjudge"
MCP_MODULE = "vizjudge.mcp.server"

# User-level skill roots scanned by common hosts. `agents` is the cross-host convention
# (DeepSeek Harness scans it without configuration); `dsh` and `claude` are host-specific.
SKILL_TARGETS = {
    "agents": Path("~/.agents/skills"),
    "dsh": Path("~/.dsh/skills"),
    "claude": Path("~/.claude/skills"),
}
PROJECT_SKILL_TARGETS = {
    "agents": Path(".agents/skills"),
    "dsh": Path(".dsh/skills"),
    "claude": Path(".claude/skills"),
}
MCP_CLIENTS = ("dsh", "json", "claude")


def skill_source() -> Path:
    """Return the skill folder: packaged copy in a wheel, repository copy in a checkout."""
    packaged = files("vizjudge") / "skill"
    if packaged.is_dir() and (packaged / "SKILL.md").is_file():
        with as_file(packaged) as path:
            return Path(path)
    repository = Path(__file__).resolve().parents[2] / "skills" / SKILL_NAME
    if (repository / "SKILL.md").is_file():
        return repository
    raise FileNotFoundError("The VizJudge skill is not bundled with this installation")


def skill_root(target: str = "agents", project: bool = False, dest: Path | None = None) -> Path:
    if dest is not None:
        return dest.expanduser()
    targets = PROJECT_SKILL_TARGETS if project else SKILL_TARGETS
    if target not in targets:
        raise ValueError(f"Unknown skill target: {target}; choose from {', '.join(targets)}")
    return targets[target].expanduser()


def install_skill(root: Path, force: bool = False) -> Path:
    """Copy the skill into ``root/vizjudge`` and return the installed folder."""
    source = skill_source()
    destination = root / SKILL_NAME
    if destination.exists():
        if not force:
            raise FileExistsError(f"{destination} already exists; pass --force to replace it")
        if not (destination / "SKILL.md").is_file():
            raise ValueError(f"Refusing to replace {destination}: it is not a skill folder")
        shutil.rmtree(destination)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__"))
    return destination


def _server_command() -> tuple[str, list[str]]:
    # The absolute interpreter avoids PATH, venv activation and uv differences across hosts.
    return sys.executable, ["-m", MCP_MODULE]


SERVER_ENV = {"MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def mcp_config(client: str) -> str:
    """Return a ready-to-paste MCP server registration for ``client``."""
    command, args = _server_command()
    if client == "dsh":
        lines = [
            "# DeepSeek Harness: add to the active profile's cordis.patch.yml, then restart dsh.",
            "- insert:",
            "    - id: mcp-vizjudge",
            "      name: '@deepseek-ai/dsh-mcp-client'",
            "      config:",
            f"        serverName: {MCP_SERVER_NAME}",
            "        transport: stdio",
            f"        command: {_yaml_quote(command)}",
            f"        args: [{', '.join(_yaml_quote(arg) for arg in args)}]",
            "        toolCallTimeoutMs: 180000",
            "        env:",
            *(f"          {key}: {_yaml_quote(value)}" for key, value in SERVER_ENV.items()),
        ]
        return "\n".join(lines) + "\n"
    if client == "json":
        server = {"command": command, "args": args, "env": SERVER_ENV}
        return json.dumps({"mcpServers": {MCP_SERVER_NAME: server}}, indent=2) + "\n"
    if client == "claude":
        quoted = f'"{command}"' if " " in command else command
        return f"claude mcp add {MCP_SERVER_NAME} -- {quoted} {' '.join(args)}\n"
    raise ValueError(f"Unknown MCP client: {client}; choose from {', '.join(MCP_CLIENTS)}")


def doctor() -> list[tuple[str, bool, str]]:
    """Check what a host agent needs; each row is (check, ok, detail)."""
    checks: list[tuple[str, bool, str]] = [
        ("vizjudge", True, f"{__version__} at {Path(__file__).resolve().parent}"),
        ("python", sys.version_info >= (3, 11), sys.executable),
    ]
    for module, hint in (
        ("mcp", 'needed for the MCP server: pip install "vizjudge[mcp]"'),
        ("pyarrow", 'optional, for Parquet: pip install "vizjudge[parquet]"'),
    ):
        found = importlib.util.find_spec(module) is not None
        name = f"{module} (optional)" if module == "pyarrow" else module
        checks.append((name, found or module == "pyarrow", "installed" if found else hint))
    try:
        importlib.import_module(MCP_MODULE)
        checks.append(("mcp server", True, f"python -m {MCP_MODULE}"))
    except Exception as exc:  # noqa: BLE001 - any import failure means the server cannot start
        checks.append(("mcp server", False, f"{MCP_MODULE} failed to import: {exc}"))
    try:
        bundled = (skill_source() / "SKILL.md").read_text(encoding="utf-8")
        checks.append(("bundled skill", True, str(skill_source())))
    except FileNotFoundError as exc:
        checks.append(("bundled skill", False, str(exc)))
        return checks
    roots = [
        *SKILL_TARGETS.values(),
        *(Path.cwd() / root for root in PROJECT_SKILL_TARGETS.values()),
    ]
    current, stale = [], []
    for root in roots:
        installed = root.expanduser() / SKILL_NAME / "SKILL.md"
        if installed.is_file():
            same = installed.read_text(encoding="utf-8") == bundled
            (current if same else stale).append(str(installed.parent))
    if current:
        detail = ", ".join(current)
        if stale:
            detail += f" (outdated copies: {', '.join(stale)})"
        checks.append(("installed skill", True, detail))
    elif stale:
        checks.append(
            (
                "installed skill",
                False,
                f"outdated at {', '.join(stale)}; run `vizjudge skill install --force`",
            )
        )
    else:
        checks.append(("installed skill", False, "not found; run `vizjudge skill install`"))
    return checks
