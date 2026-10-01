# Upstream repositories (not vendored)

Clones used as read-only references during the campaign, pinned to the commit
that was checked out locally:

| repo | commit | role |
|---|---|---|
| https://github.com/aaronw5/PHAT-JeT | `28f4e5960e3bc32ed8612921f165998ce1b825b6` | upstream model; NOT synthesizable, do not port verbatim (see CLAUDE.md) |
| https://github.com/calad0i/HGQ2-examples | `6cdc6e34d4cea0fb0f2e3ac502e2bfea01b2a805` | `jsc150/` is the reference QAT recipe |
| https://github.com/calad0i/JEDI-linear | `48032802068bf1e4878b5b119a6e8a87512c733e` | comparison model; ported as `build_jedi_linear` in `phat_jet_k3.py` (D120) |

Reference paper for the comparison and the CTL2 constraints: arXiv:2508.15468.
