# Vendored: rsl_rl

- Upstream: https://github.com/leggedrobotics/rsl_rl
- Version: v5.4.1 (commit 016c7ed)
- License: BSD-3-Clause (see LICENSE)

Vendored so the library can be modified in-repo alongside the SO-101 RL
examples (`newton/examples/rl/`), which prepend `newton/_src/rl` to
`sys.path` so `import rsl_rl` resolves to this copy. Runtime deps not
vendored: torch, tensordict, numpy, gitpython, tensorboard.
