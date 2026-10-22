import json
import sys

import pytest

from vizjudge.cli import main
from vizjudge.integrations import install_skill, mcp_config, skill_source


def test_bundled_skill_has_host_frontmatter():
    text = (skill_source() / "SKILL.md").read_text(encoding="utf-8")
    header = text.split("---")[1]
    fields = dict(line.split(": ", 1) for line in header.strip().splitlines())
    assert fields["name"] == "vizjudge"
    # DeepSeek Harness truncates catalog descriptions at 500 characters.
    assert 0 < len(fields["description"]) <= 500


def test_skill_install_copies_and_refuses_overwrite(tmp_path, capsys):
    assert main(["skill", "install", "--dest", str(tmp_path)]) == 0
    installed = tmp_path / "vizjudge"
    assert (installed / "SKILL.md").is_file()
    assert (installed / "agents" / "openai.yaml").is_file()
    assert "Installed the VizJudge skill" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        main(["skill", "install", "--dest", str(tmp_path)])
    (installed / "SKILL.md").write_text("stale", encoding="utf-8")
    assert main(["skill", "install", "--dest", str(tmp_path), "--force"]) == 0
    assert (installed / "SKILL.md").read_text(encoding="utf-8").startswith("---")


def test_force_install_refuses_to_delete_non_skill_folder(tmp_path):
    (tmp_path / "vizjudge").mkdir()
    (tmp_path / "vizjudge" / "notes.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="not a skill folder"):
        install_skill(tmp_path, force=True)
    assert (tmp_path / "vizjudge" / "notes.txt").is_file()


def test_project_install_uses_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["skill", "install", "--target", "dsh", "--project"]) == 0
    assert (tmp_path / ".dsh" / "skills" / "vizjudge" / "SKILL.md").is_file()


def test_mcp_config_targets_this_interpreter(capsys):
    assert main(["mcp-config", "--client", "dsh"]) == 0
    dsh = capsys.readouterr().out
    assert "'@deepseek-ai/dsh-mcp-client'" in dsh
    assert "serverName: vizjudge" in dsh
    assert "transport: stdio" in dsh
    assert "'" + sys.executable.replace("'", "''") + "'" in dsh
    assert "'vizjudge.mcp.server'" in dsh

    server = json.loads(mcp_config("json"))["mcpServers"]["vizjudge"]
    assert server["command"] == sys.executable
    assert server["args"] == ["-m", "vizjudge.mcp.server"]
    assert mcp_config("claude").startswith("claude mcp add vizjudge -- ")


def test_doctor_reports_checks(capsys):
    code = main(["doctor"])
    output = capsys.readouterr().out
    assert "[ok] vizjudge:" in output
    assert "bundled skill" in output
    assert code in {0, 1}


def test_doctor_fails_until_a_current_skill_is_installed(tmp_path, monkeypatch):
    from vizjudge import integrations

    user_root = tmp_path / "user"
    monkeypatch.setattr(integrations, "SKILL_TARGETS", {"agents": user_root})
    monkeypatch.setattr(integrations, "PROJECT_SKILL_TARGETS", {})

    def skill_row():
        return next(row for row in integrations.doctor() if row[0] == "installed skill")

    assert skill_row()[1] is False
    installed = install_skill(user_root)
    assert skill_row()[1] is True
    (installed / "SKILL.md").write_text("stale copy", encoding="utf-8")
    ok, detail = skill_row()[1:]
    assert ok is False and "--force" in detail
