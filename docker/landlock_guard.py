#!/usr/bin/env python3
"""Exec a command under a Landlock ruleset that denies file/directory removal
(and re-linking/renaming) everywhere except the paths listed in the
LANDLOCK_ALLOW_REMOVE env var (colon-separated).

Creating, writing and truncating files stays unrestricted; combined with the
seccomp profile that no-ops chmod, code under test can never delete, rename or
re-protect anything in the mounted result directories. Fails closed: if
Landlock is unavailable the command is not run at all.
"""

import ctypes
import os
import sys

libc = ctypes.CDLL(None, use_errno=True)

SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1

ACCESS_FS_REMOVE_DIR = 1 << 4
ACCESS_FS_REMOVE_FILE = 1 << 5
ACCESS_FS_REFER = 1 << 13  # ABI >= 2: cross-directory rename/link

PR_SET_NO_NEW_PRIVS = 38


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def die(msg: str):
    print(f"[landlock_guard] FATAL: {msg}", file=sys.stderr)
    sys.exit(70)


def main():
    if len(sys.argv) < 2:
        die("usage: landlock_guard.py <command> [args...]")

    abi = libc.syscall(SYS_LANDLOCK_CREATE_RULESET, None, 0, LANDLOCK_CREATE_RULESET_VERSION)
    if abi < 0:
        die(f"Landlock unsupported by kernel (errno {ctypes.get_errno()})")

    access = ACCESS_FS_REMOVE_DIR | ACCESS_FS_REMOVE_FILE
    if abi >= 2:
        # Without REFER, cross-directory renames are denied everywhere, even in
        # allowed paths; handle + grant it so scratch dirs keep full semantics.
        access |= ACCESS_FS_REFER

    attr = RulesetAttr(access)
    ruleset_fd = libc.syscall(
        SYS_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0
    )
    if ruleset_fd < 0:
        die(f"landlock_create_ruleset failed (errno {ctypes.get_errno()})")

    for path in filter(None, os.environ.get("LANDLOCK_ALLOW_REMOVE", "").split(":")):
        try:
            parent_fd = os.open(path, os.O_PATH)
        except FileNotFoundError:
            # a missing scratch path just stays non-removable; not a safety issue
            print(f"[landlock_guard] skipping missing allowed path {path!r}", file=sys.stderr)
            continue
        except OSError as e:
            die(f"cannot open allowed path {path!r}: {e}")
        rule = PathBeneathAttr(access, parent_fd)
        if (
            libc.syscall(
                SYS_LANDLOCK_ADD_RULE, ruleset_fd, LANDLOCK_RULE_PATH_BENEATH, ctypes.byref(rule), 0
            )
            < 0
        ):
            die(f"landlock_add_rule failed for {path!r} (errno {ctypes.get_errno()})")
        os.close(parent_fd)

    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0:
        die(f"prctl(NO_NEW_PRIVS) failed (errno {ctypes.get_errno()})")
    if libc.syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0) < 0:
        die(f"landlock_restrict_self failed (errno {ctypes.get_errno()})")
    os.close(ruleset_fd)

    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
