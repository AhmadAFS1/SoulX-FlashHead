#!/usr/bin/env bash
# Verify + decrypt + load the delivered avatar-service image.
#
# Downloads the encrypted image, checks its SHA-256 BEFORE using it, then decrypts and loads
# it into Docker. The passphrase is prompted for (we send it separately from the URL — that
# split is deliberate: neither one alone is enough to open the image).
#
# Usage:
#   ./scripts/load_image.sh <download-url> <expected-sha256>
#   ./scripts/load_image.sh ./kit-1.0.tar.gz.gpg <expected-sha256>   # already downloaded
#
# Passphrase: by default gpg prompts for it interactively (we send it separately from the URL).
# For non-interactive use (CI, provisioning, Packer/Ansible) set OJIN_PASSPHRASE in the environment:
#   OJIN_PASSPHRASE='…' ./scripts/load_image.sh <url-or-file> <sha256>
#
# Works fully offline with a local file — copy the .gpg onto an air-gapped machine and pass
# its path instead of a URL.
#
# Requires: curl, sha256sum, gpg (2.x), docker.
set -euo pipefail

# Point gpg's pinentry at the current terminal for the interactive prompt (standard practice; a
# no-op when there is no tty). Without it, some environments fail with a misleading "problem with
# the agent: Required environment variable not set" that looks like a bad passphrase.
export GPG_TTY="$(tty 2>/dev/null || true)"

SRC="${1:?usage: load_image.sh <download-url-or-file> <expected-sha256>}"
EXPECTED_SHA="${2:?missing <expected-sha256>}"

command -v gpg >/dev/null || { echo "gpg not found" >&2; exit 1; }
command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }

ENC="ojin-avatar-service.tar.gz.gpg"

if [[ "$SRC" == http* ]]; then
  echo "[1/3] downloading (resumable — safe to re-run if interrupted)…" >&2
  # -C - resumes a partial file rather than starting the multi-GB download over. If the file is
  # ALREADY complete, S3 answers 416 (Range Not Satisfiable) and curl exits non-zero (33/22) — that
  # is not an error here: fall through to the SHA-256 check, which is the authoritative test of
  # completeness/integrity. Any other curl failure is fatal.
  rc=0
  curl -fL -C - -o "$ENC" "$SRC" || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    if [[ ( "$rc" -eq 33 || "$rc" -eq 22 ) && -s "$ENC" ]]; then
      echo "  (server reports nothing left to fetch — verifying the existing file)" >&2
    else
      echo "download failed (curl exit $rc)" >&2
      exit "$rc"
    fi
  fi
else
  [[ -f "$SRC" ]] || { echo "no such file: $SRC" >&2; exit 1; }
  ENC="$SRC"
  echo "[1/3] using local file: $ENC" >&2
fi

echo "[2/3] verifying SHA-256…" >&2
ACTUAL_SHA="$(sha256sum "$ENC" | awk '{print $1}')"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "CHECKSUM MISMATCH — do NOT load this file." >&2
  echo "  expected: $EXPECTED_SHA" >&2
  echo "  actual:   $ACTUAL_SHA" >&2
  echo "The download is incomplete or the file was altered; re-download and try again." >&2
  exit 1
fi
echo "  OK: $ACTUAL_SHA" >&2

# Streamed: decrypt -> decompress -> docker load, so no ~20 GB plaintext tarball hits disk.
if [[ -n "${OJIN_PASSPHRASE:-}" ]]; then
  echo "[3/3] decrypt + load (passphrase from OJIN_PASSPHRASE):" >&2
  gpg --batch --pinentry-mode loopback --passphrase "$OJIN_PASSPHRASE" --decrypt "$ENC" | gunzip | docker load
else
  echo "[3/3] decrypt + load (enter the passphrase we sent separately):" >&2
  gpg --decrypt "$ENC" | gunzip | docker load
fi

cat >&2 <<'EOF'

Loaded. Next:
    docker compose --profile demo up      # -> out/avatar.mp4
EOF
