"""verify_reproducibility.py --run <id> --repeat 2 --report <path>

Two separate checks on a published run: (a) re-scoring with the run's SAVED models must give the
stored checksum; (b) re-training from the same inputs must give it too. Drift in data, code, config files or
saved models is reported first and stops the comparison (a new run, or the recorded commit, is required)."""
from __future__ import annotations

from _common import finish, main_guard, parser, run_dir, settings

from pharmacy_potential.reproducibility import verify_run


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--repeat", type=int, default=2)
    a = p.parse_args()
    s = settings(a)
    run_dir(s, a.run)
    finish(a.report, "verify_reproducibility", verify_run(s, a.run, a.repeat), {"run_id": a.run})


if __name__ == "__main__":
    main_guard(main)
