"""Filesystem paths shared by host and harness-process lifecycle code."""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path

from omnigent._platform import IS_WINDOWS

HARNESS_TMP_PARENT_ENV_VAR = "OMNIGENT_HARNESS_TMP_PARENT"


def harness_tmp_parent(env: Mapping[str, str] | None = None) -> Path:
    """Return the configured harness socket root without changing relativity."""
    source = os.environ if env is None else env
    configured = source.get(HARNESS_TMP_PARENT_ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    if IS_WINDOWS:
        return Path(tempfile.gettempdir()) / "omnigent"
    return Path(f"/tmp/omnigent-{os.getuid()}")


def absolute_harness_tmp_parent(path: Path) -> Path:
    """Make a harness socket root absolute without resolving symlinks."""
    return Path(os.path.abspath(path.expanduser()))


def resolve_harness_tmp_parent(env: Mapping[str, str] | None = None) -> Path:
    """Return one short absolute socket root for cross-process propagation."""
    return absolute_harness_tmp_parent(harness_tmp_parent(env))


def ensure_private_dir(path: Path, my_uid: int | None) -> None:
    """
    Create ``path`` as a 0o700 directory, or validate an existing one.

    :param path: Directory that must be owner-only, e.g. one bridge ancestor
        or the harness socket root.
    :param my_uid: Current uid, or ``None`` where POSIX ownership does not
        apply (Windows), which skips the owner and mode checks.
    :raises RuntimeError: If ``path`` exists as a symlink, a non-directory,
        or a directory owned by another uid.
    """
    try:
        os.mkdir(path, mode=0o700)
        return
    except FileExistsError:
        pass
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise RuntimeError(f"refusing to use {path!s}: is a symlink")
    if not stat.S_ISDIR(st.st_mode):
        raise RuntimeError(f"refusing to use {path!s}: not a directory")
    if my_uid is not None and st.st_uid != my_uid:
        raise RuntimeError(
            f"refusing to use {path!s}: owned by uid {st.st_uid}, not current user ({my_uid})"
        )
    if my_uid is not None and (st.st_mode & 0o077) != 0:
        os.chmod(path, 0o700)
