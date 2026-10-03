# Working on omnibias

Read the [capability rule](.cursor/rules/omnibias.mdc) once, unless already loaded.
For maintenance, load only `.agents/skills/omnibias-<package>/SKILL.md` for the
affected primitive. Each skill owns its package's implementation context.
Consumer repositories own their own skills under `../omnibias_projects/`.

[CONTRIBUTING.md](CONTRIBUTING.md) owns development commands and contribution
policy. [Package metadata](docs/packages.md) and [API pages](docs/api/core.md)
are generated; do not repeat their inventories in agent guidance.

Before numerical edits, follow the selected skill's shared-contract link.
Do not load the whole skill catalog or historical research guidance.
