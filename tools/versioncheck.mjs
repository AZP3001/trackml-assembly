// versioncheck.mjs — proves the version number in .github/workflows/deploy.yml
// is actually computed from git history, and that the constant baked into the
// workflow (PORT_COMMIT) still points at the right place.
//
// Nothing here talks to wasm; this is pure git plumbing. It exists because the
// whole point of the versioning scheme is that nobody ever types a version
// number by hand — see deploy.yml's "Compute version" step for the reasoning
// (the predecessor project, AZP3001/main, hand-maintained a "TrackML V14.0"
// style string and has commits with subjects like "Fixxed Version number
// forgotten from last update" to show for it). A scheme that can't be
// hand-maintained also can't be hand-verified by eyeballing a diff, so this
// checks it by literally running the same commands the workflow runs.
//
//   node tools/versioncheck.mjs
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, '..');

let failures = 0;
const check = (ok, name, detail) => {
    if (!ok) failures++;
    console.log(`${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
};

function git(...args) {
    return execFileSync('git', args, { cwd: repoRoot, encoding: 'utf8' }).trim();
}

console.log('version scheme test\n');

// --- the PORT_COMMIT baked into deploy.yml really is V21.0 --------------
const workflow = readFileSync(join(repoRoot, '.github/workflows/deploy.yml'), 'utf8');
const m = workflow.match(/PORT_COMMIT:\s*([0-9a-f]{40})/);
check(!!m, 'deploy.yml names a 40-char PORT_COMMIT', m ? m[1] : 'no match found');
if (!m) { console.log(); console.error('1 version check failed.'); process.exit(1); }
const PORT_COMMIT = m[1];

let subject = '';
try { subject = git('log', '-1', '--format=%s', PORT_COMMIT); }
catch (e) { /* handled by the ancestor check below */ }
check(/webassembly|wasm/i.test(subject), 'PORT_COMMIT is the WebAssembly port commit',
    subject || '(commit not found in this checkout)');

let parents = '';
try { parents = git('log', '-1', '--format=%P', PORT_COMMIT); }
catch (e) { /* ignore */ }
check(parents === '', 'PORT_COMMIT is a root commit — this repo\'s first',
    parents === '' ? '(no parents)' : `parents: ${parents}`);

// --- the formula in deploy.yml, run here the same way -------------------
// (mirrors: git merge-base --is-ancestor "$PORT_COMMIT" HEAD;
//           git rev-list --count "${PORT_COMMIT}..HEAD")
let isAncestor = false;
try {
    execFileSync('git', ['merge-base', '--is-ancestor', PORT_COMMIT, 'HEAD'], { cwd: repoRoot });
    isAncestor = true;
} catch (e) { isAncestor = false; }
check(isAncestor, 'PORT_COMMIT is an ancestor of HEAD',
    isAncestor ? 'ancestry holds' : 'HEAD has diverged from PORT_COMMIT — the workflow would fail loudly, correctly');

if (isAncestor) {
    const since = Number(git('rev-list', '--count', `${PORT_COMMIT}..HEAD`));
    const version = `21.${since}`;
    check(Number.isInteger(since) && since >= 0, 'commit count since the port is a sane non-negative integer',
        `${since} commits since PORT_COMMIT`);
    console.log(`  HEAD would deploy as: V${version}`);

    // --- monotonicity: version must never go backwards on the default branch
    // A history rewrite could, in principle, make HEAD's ancestor count
    // smaller than an earlier deploy's. What actually got deployed over time
    // is the FIRST-PARENT chain — the sequence HEAD itself moved through,
    // whether by a plain commit or a merge landing a side branch — not every
    // commit reachable from HEAD. Walking the full DAG instead of just that
    // chain used to fail this the moment two branches (kept identical until
    // now by convention) genuinely diverged and got merged back together: a
    // commit that only ever existed on the side branch has a smaller count
    // than commits made in parallel on this one, even though neither was ever
    // individually deployed on its own — only the merge that combined them
    // was. Each link in --first-parent, by contrast, actually is an ancestor
    // of the next, so its count can only stay the same or grow.
    const recentLog = git('log', '--first-parent', '--reverse', '--format=%H', `${PORT_COMMIT}..HEAD`).split('\n').filter(Boolean);
    let prevCount = 0, monotone = true, brokenAt = null;
    for (const sha of recentLog) {
        const c = Number(git('rev-list', '--count', `${PORT_COMMIT}..${sha}`));
        if (c < prevCount) { monotone = false; brokenAt = sha; break; }
        prevCount = c;
    }
    check(monotone, 'the version number never decreases walking the branch forward',
        monotone ? `checked ${recentLog.length} commits since the port` : `regressed at ${brokenAt}`);
}

// --- the workflow actually wires the computed value into version.json ---
check(/"version":\s*"\$\{VERSION\}"/.test(workflow),
    'the computed version is written into version.json',
    /"version":\s*"\$\{VERSION\}"/.test(workflow) ? 'found in the Stamp step' : 'not found — check the heredoc');
check(/fetch-depth:\s*0/.test(workflow),
    'the deploy checkout fetches full history',
    'required for git rev-list to see more than the latest commit');

// --- the frontend actually renders a "version" field if given one -------
const scriptJs = readFileSync(join(repoRoot, 'script.js'), 'utf8');
check(/v\.version/.test(scriptJs), 'script.js reads the version field from version.json',
    '(showBuildStamp)');
check(/version-tag-compact/.test(scriptJs) && /version-tag-compact/.test(readFileSync(join(repoRoot, 'index.html'), 'utf8')),
    'the compact mobile version spans exist in both script.js and index.html');

console.log();
if (failures) { console.error(`${failures} version check(s) failed.`); process.exit(1); }
console.log('the version number is derived from git history, not hand-typed.');
