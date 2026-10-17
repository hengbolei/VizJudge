# Using VizJudge in DeepSeek Harness

VizJudge plugs into [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
(`dsh`) as two pieces: an **MCP server** that supplies the statistical evidence, and an
**Agent Skill** that teaches the DeepSeek model to act as the Judge. Both come from
the same Python package.

> DeepSeek Harness is in developer preview, and its configuration format may change
> between releases. The MCP patch below follows the `@deepseek-ai/dsh-mcp-client`
> README. If your version rejects it, compare it with `dsh web --dump-config`.

## 1. Install VizJudge

Use a dedicated environment. `dsh` launches the server with the absolute interpreter
path, so you don't need to activate it later.

```bash
python -m venv ~/.vizjudge
~/.vizjudge/bin/pip install "vizjudge[mcp] @ git+https://github.com/hengbolei/VizJudge"
```

On Windows, use `~/.vizjudge/Scripts/pip.exe`. Once VizJudge is published on PyPI,
`pip install "vizjudge[mcp]"` is enough. From a checkout, `uv sync --extra mcp` also works.

## 2. Install the skill

```bash
vizjudge skill install                    # ~/.agents/skills/vizjudge (scanned by dsh)
vizjudge skill install --target dsh       # ~/.dsh/skills/vizjudge
vizjudge skill install --project          # ./.agents/skills/vizjudge for one project
```

`dsh` discovers `<root>/<name>/SKILL.md` bundles in project `.dsh/skills` and
`.agents/skills`, then user `<dshHome>/skills` and `<agentsHome>/skills`. A project
copy wins over a user copy. Pass `--force` to update an installed copy.

## 3. Register the MCP server

Print a patch for this exact interpreter:

```bash
vizjudge mcp-config --client dsh
```

```yaml
- insert:
    - id: mcp-vizjudge
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: vizjudge
        transport: stdio
        command: '/home/you/.vizjudge/bin/python'
        args: ['-m', 'vizjudge.mcp.server']
        toolCallTimeoutMs: 180000
        env:
          MPLBACKEND: 'Agg'
          PYTHONIOENCODING: 'utf-8'
```

Append it to the active profile's patch file (for the web profile,
`~/.dsh/profiles/web/cordis.patch.yml`), then restart `dsh`:

```bash
npx @deepseek-ai/dsh web
```

The tools appear as `mcp__vizjudge__profile_dataset`, `mcp__vizjudge__generate_candidates`,
`mcp__vizjudge__validate_features`, `mcp__vizjudge__select_charts`,
`mcp__vizjudge__render_charts`, `mcp__vizjudge__analyze_dataset` and
`mcp__vizjudge__run_agent`.

Notes:

- `toolCallTimeoutMs` is raised from the 60 s default because cross-validation on
  larger files can take longer. VizJudge's own `budget` limits still apply.
- `dsh` removes ambient variables whose names contain `KEY`, `TOKEN`, `SECRET` or
  `PASSWORD` before starting stdio servers. VizJudge needs none of them. Only the
  optional HTTP Judge of `run_agent` reads `VIZJUDGE_JUDGE_TOKEN`. To use it, add the
  variable to `env` explicitly.
- `vizjudge doctor` checks the interpreter, the `mcp` dependency and installed skill copies.

## 4. Try it

Open a session in a project that contains a dataset, or use the bundled example
`examples/breast_cancer/breast_cancer.csv`, and ask:

```text
用 vizjudge 分析 <absolute path>/breast_cancer.csv，目标列是 diagnosis（分类任务）。
告诉我哪几张图最值得看、证据是什么、下一步该怎么改特征和验证方案。
```

Check that the session:

1. loads the `vizjudge` skill (catalog entry `vizjudge`);
2. calls `profile_dataset` → `generate_candidates` → `validate_features` → `select_charts`
   (once with statistics, once with its own `reviews`) → `render_charts`, passing the same
   `target`, `task` and `seed` each time;
3. cites exact evidence IDs such as `C….incremental_gain`, and never raises a
   statistical priority;
4. writes the PNG files under the absolute `output_dir` it chose.

A good leakage test: add a column copied from the target (for example
`diagnosis_code`). The model should defer that chart as leakage instead of
recommending it.

If no tools appear, the skill tells the model to fall back to the CLI
(`vizjudge agent … --output …`) and to show you the `mcp-config` output.
