#!/usr/bin/env python3
"""Download and restore the Wizard pre-encoded training data into one root folder (the $WIZARD_DATA of the training code).

    python restore_precoded.py --root /path/to/wizard_data                    # download + extract + fix paths
    python restore_precoded.py --root /path/to/wizard_data --verify           # also check every file's sha256 against the manifests
    python restore_precoded.py --root /path/to/wizard_data --dirs windows,latents --keep-tars   # a subset, keep the downloaded shards

What it does:
  1. downloads precoded/<dir>/shard-*.tar(.tsv) and precoded_meta/ from yuzhewu207/wizard-precoded (huggingface_hub; resumable);
  2. extracts every shard into --root member by member. The shards were packed with the folder names used when the data was built;
     every member path is mapped to the released layout on extraction (NAME_MAP below, e.g. <build name of windows>/<proxy>/clip_<key>.pt
     -> windows/proxy/clip_<key>.pt), and symlink targets are mapped the same way;
  3. copies precoded_meta/* over it (row files and text tables without the WildWorld rows);
  4. the data was built under BUILD_ROOT: absolute symlinks (samples/clips/*) and the paths inside the row files (*.jsonl, *.json, *.txt at
     the top of each folder) are rewritten to --root and to the released folder names;
  5. --verify: sha256 of every extracted file against the shard manifests.
Needs ~0.85 TB free (+ one shard while it is downloaded and extracted; each tar is deleted after extraction unless --keep-tars).
With access to the private part (WildWorld windows) pass --private-repo yuzhewu207/wizard-repro: its v2/precoded_ww shards and the full row files are added.
Running it again is safe: finished shards are skipped and paths that already use the released names are left as they are.
"""
import argparse, hashlib, os, re, sys, tarfile, glob, shutil

BUILD_ROOT = "/data/yuzhewu/datasets"          # where the data was built: absolute paths in the shards start with it

# Folder names at build time -> released names (paths relative to the data root). The longest matching prefix (whole path
# components) wins. None = not restored (files that are not released, or a link replaced by a real folder).
NAME_MAP = {
    "wan_v7": "windows",
    "wan_v7/cond7": "windows/proxy",
    "wan_v7/cond7_unk2": "windows/proxy_unk2",
    "wan_v7/cond7_trueunk": "windows/proxy_trueunk",
    "wan_v7/cond7_memunk": "windows/proxy_memunk",
    "wan_v7/gbuf2": "windows/geometry",                # a relative link to ../geometry
    "wan_v7/shield": None,                             # was a link to wan_v6/shield; that folder itself becomes windows/shield
    "wan_v7/v7_rows.jsonl": "windows/rows.jsonl",
    "wan_v7/v5_list.jsonl": None,
    "wan_v7/v6_rows.jsonl": None,
    "wan_v6/gbuf2": "geometry",
    "wan_v6/shield": "windows/shield",
    "wan_v6": None,                                    # nothing else of it is released
    "wan_v6_gbuf2": "geometry",                        # (repository folder name of the geometry shards, for --dirs)
    "wan_mix": "latents",
    "wan_mix5": "samples",
    "wan_entry_v3": "synth_entry",
    "wan_entry_v3x": "synth_entry_orbit",
    "wan_turn_g": "synth_turn",
    "wan_lookback": "synth_lookback",
    "wan_lookback_geo": "synth_lookback_geo",
    "wan_roam_unarmed": "roam_unarmed",
    "wan_memroam": "roam_memory",
    "wan_sfly": "scenefly",
    "wan_sfly_replay": "scenefly_replay",
    "wan_sfly_replay_test": "scenefly_replay_test",
    "wan_wedit": "worldedit",
}
# idempotent on a partly restored tree: the build-time sub-folder names under the released top folder map the same way
NAME_MAP.update({"windows" + k[len("wan_v7"):]: v for k, v in list(NAME_MAP.items()) if k.startswith("wan_v7/") and k != "wan_v7/shield"})
# targets of links / paths in row files: a path into a folder that is not restored as such still has a released home
TARGET_MAP = {"wan_v7/shield": "windows/shield"}
_OLD_TOPS = sorted({k.split("/")[0] for k in NAME_MAP}, key=len, reverse=True)
_TOP = r"(" + "|".join(map(re.escape, _OLD_TOPS)) + r")(?=[/\"'\s]|$)([^\"'\s,]*)"
_TEXT_PAT = re.compile(r"(" + re.escape(BUILD_ROOT) + r"/)(?:" + _TOP + r")?|(?<=[\"'/\s])" + _TOP)


def _lookup(table, p):
    """longest key of table that is p or a leading part of p (whole components) -> (key, value) or (None, None)"""
    parts = p.split("/")
    for n in range(len(parts), 0, -1):
        k = "/".join(parts[:n])
        if k in table: return k, table[k]
    return None, None


def map_path(p, member=True):
    """a path relative to the data root (build or released names) -> released path; None if it is not restored (member=True).
    member=False maps a link target / a path inside a row file: never None (an unknown folder keeps its name)."""
    p = p.lstrip("/")
    while p.startswith("./"): p = p[2:]
    if not member:
        k, v = _lookup(TARGET_MAP, p)
        if k is not None: return v + p[len(k):]
    k, v = _lookup(NAME_MAP, p)
    if k is None: return p                                     # released name already, or a folder the map does not know
    if v is None: return None if member else p
    return v + p[len(k):]


def map_link(target, member_old, root):
    """symlink target of the build-time member member_old -> target for the restored link. Absolute targets under BUILD_ROOT go to
    root (absolute), absolute targets already under root are re-mapped there; relative targets stay relative to the restored link."""
    if os.path.isabs(target):
        for base in (BUILD_ROOT, root):
            if target == base or target.startswith(base + "/"):
                return os.path.join(root, map_path(target[len(base) + 1:], member=False))
        return target
    old_abs = os.path.normpath(os.path.join(os.path.dirname(member_old), target))      # relative to the data root, build names
    if old_abs.startswith(".."): return target
    new_link = map_path(member_old)
    return os.path.relpath(os.path.join(root, map_path(old_abs, member=False)), os.path.dirname(os.path.join(root, new_link)))


def map_text(s, root):
    """paths inside a row / index file: BUILD_ROOT/... -> root/..., with <old>/... -> <new>/... right after it; elsewhere <old> as a path
    component (next to a '/') -> <new>"""
    def sub(m):
        if m.group(1):                                                                   # BUILD_ROOT/[<old>...]
            return root + "/" + (map_path(m.group(2) + m.group(3), member=False) if m.group(2) else "")
        top, rest = m.group(4), m.group(5)
        if not rest.startswith("/") and m.string[m.start() - 1:m.start()] != "/": return m.group(0)   # a bare word, not a path
        return map_path(top + rest, member=False)
    return _TEXT_PAT.sub(sub, s)


def sha256(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()


def _inside(root, rel):
    q = os.path.normpath(os.path.join(root, rel)); return q == root or q.startswith(root + os.sep)


def _clear(q):
    if os.path.islink(q) or os.path.isfile(q): os.remove(q)
    elif os.path.isdir(q): shutil.rmtree(q)


def extract(tar_path, root):
    """extract one shard member by member under the released names -> (written, skipped)"""
    n = sk = 0
    with tarfile.open(tar_path) as tf:
        for m in tf:
            old = m.name.lstrip("/")
            while old.startswith("./"): old = old[2:]
            new = map_path(old)
            if not new or not _inside(root, new): sk += 1; continue
            q = os.path.join(root, new)
            if m.isdir(): os.makedirs(q, exist_ok=True); continue
            os.makedirs(os.path.dirname(q), exist_ok=True)
            if m.issym():
                if os.path.lexists(q): _clear(q)
                os.symlink(map_link(m.linkname, old, root), q)
            elif m.islnk():
                src = map_path(m.linkname.lstrip("/"))
                if not src or not _inside(root, src): sk += 1; continue
                if os.path.lexists(q): _clear(q)
                os.link(os.path.join(root, src), q)
            elif m.isfile():
                if os.path.lexists(q) and not os.path.isfile(q) or os.path.islink(q): _clear(q)
                with tf.extractfile(m) as src, open(q, "wb") as dst: shutil.copyfileobj(src, dst, 1 << 22)
                os.chmod(q, m.mode & 0o755); os.utime(q, (m.mtime, m.mtime))
            else: sk += 1; continue
            n += 1
    return n, sk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--repo", default="yuzhewu207/wizard-precoded")
    ap.add_argument("--private-repo", default=""); ap.add_argument("--dirs", default="", help="comma list of folders, released names (default: all)")
    ap.add_argument("--dl", default="", help="download dir (default: <root>/_download)"); ap.add_argument("--keep-tars", action="store_true")
    ap.add_argument("--verify", action="store_true"); ap.add_argument("--local", default="", help="use already-downloaded repo folder instead of downloading")
    a = ap.parse_args()
    root = os.path.abspath(a.root); os.makedirs(root, exist_ok=True); dl = a.local or a.dl or os.path.join(root, "_download")
    want = set((map_path(x, member=False) or x).split("/")[0] for x in a.dirs.split(",") if x)
    if not a.local: from huggingface_hub import HfApi, hf_hub_download
    srcs = [(a.repo, "precoded/", "precoded_meta/")] + ([(a.private_repo, "v2/precoded_ww/", "v2/precoded_meta_full/")] if a.private_repo else [])
    for repo, prefix, meta in srcs:
        files = HfApi().list_repo_files(repo, repo_type="dataset") if not a.local else \
            [os.path.relpath(p, dl) for p in glob.glob(f"{dl}/{prefix}**/*", recursive=True) + glob.glob(f"{dl}/{meta}**/*", recursive=True) if os.path.isfile(p)]
        tars = sorted(f for f in files if f.startswith(prefix) and f.endswith(".tar") and (not want or map_path(f.split("/")[-2], member=False) in want))
        print(f"{repo}: {len(tars)} shards", flush=True)
        for i, t in enumerate(tars):
            mark = os.path.join(root, "_restored", t.replace("/", "__") + ".done")
            if os.path.exists(mark): continue
            p = os.path.join(dl, t) if a.local else hf_hub_download(repo, t, repo_type="dataset", local_dir=dl)
            m = os.path.join(dl, t + ".tsv") if a.local else hf_hub_download(repo, t + ".tsv", repo_type="dataset", local_dir=dl)
            n, sk = extract(p, root)
            os.makedirs(os.path.dirname(mark), exist_ok=True); shutil.copy(m, mark)
            if not a.keep_tars and not a.local: os.remove(p)
            print(f"  [{i + 1}/{len(tars)}] {t}: {n} entries" + (f", {sk} not restored" if sk else ""), flush=True)
        metas = [f for f in files if f.startswith(meta)]
        nm = 0
        for f in metas:
            rel = map_path(f[len(meta):])
            if not rel: continue
            p = os.path.join(dl, f) if a.local else hf_hub_download(repo, f, repo_type="dataset", local_dir=dl)
            dst = os.path.join(root, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.islink(dst): os.remove(dst)
            shutil.copy(p, dst); nm += 1
        print(f"  meta: {nm} files from {meta}", flush=True)
    # absolute link targets left from the build (e.g. from an earlier restore) and the paths inside the row / index files -> root, released names
    nl = nt = 0
    for d in sorted(os.listdir(root)):
        dd = os.path.join(root, d)
        if not os.path.isdir(dd) or os.path.islink(dd) or d.startswith("_"): continue
        for base, dirs, fs in os.walk(dd):
            for x in fs + dirs:
                q = os.path.join(base, x)
                if os.path.islink(q) and os.readlink(q).startswith(BUILD_ROOT + "/"):
                    t = map_link(os.readlink(q), os.path.relpath(q, root), root); os.remove(q); os.symlink(t, q); nl += 1
        for f in glob.glob(f"{dd}/*.jsonl") + glob.glob(f"{dd}/*.json") + glob.glob(f"{dd}/*.txt"):
            s = open(f, encoding="utf-8").read(); s2 = map_text(s, root)
            if s2 != s: open(f, "w", encoding="utf-8").write(s2); nt += 1
    print(f"rewrote {nl} symlinks and {nt} row/index files to {root}", flush=True)
    if a.verify:
        bad = n = rw = 0
        for m in glob.glob(os.path.join(root, "_restored", "*.done")):
            for line in open(m):
                size, h, rel = line.rstrip("\n").split("\t", 2)
                if h.startswith("symlink->"): continue
                rel = map_path(rel)                                  # the manifests list released paths; build-time ones are mapped too
                if not rel: continue
                if rel.count("/") == 1 and rel.endswith((".jsonl", ".json", ".txt")): rw += 1; continue   # row / index files: paths rewritten in step 4
                n += 1; p = os.path.join(root, rel)
                if not os.path.exists(p) or sha256(p) != h: bad += 1; print("BAD", rel)
        print(f"verified {n} files, {bad} bad ({rw} row/index files not compared: their paths were rewritten)")
    dangling = [q for q in glob.glob(f"{root}/samples/clips/*") if os.path.islink(q) and not os.path.exists(q)]
    print(f"dangling samples/clips links: {len(dangling)}" + (" (expected: the WildWorld clips are not in the public part)" if dangling and not a.private_repo else ""))
    print(f"done. export WIZARD_DATA={root}")


if __name__ == "__main__":
    main()
