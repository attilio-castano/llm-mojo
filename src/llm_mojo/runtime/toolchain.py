"""Read-only checks for the tools that setup, preparation and native builds need.

Each check reports what it found and the command that fixes it. Nothing here
changes the system: remedies that need administrator rights are only printed.
"""
from dataclasses import dataclass
import json
import platform
import shutil
import subprocess

from llm_mojo._repository import environment_tool

MEASURED_DEVICE = 'Apple M4 Pro'


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    remedy: str = ''
    # What cannot proceed without it: 'all', 'build', 'prepare', or '' for a note.
    blocks: str = 'build'


def run(*command, timeout=30):
    """Return (exit status, combined output); a missing tool is a failed check, not an error."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return 1, str(error)
    return result.returncode, (result.stdout + result.stderr).strip()


def platform_check(system=platform.system, machine=platform.machine):
    ok = system() == 'Darwin' and machine() == 'arm64'
    return Check('Apple Silicon Mac', ok, f'{system()} {machine()}',
                 'llm-mojo runs on Apple Silicon Macs with Metal.', 'all')


def xcode_checks(runner=run):
    """The documented Metal compiler prerequisites (docs/development.md)."""
    checks = []
    status, path = runner('xcode-select', '-p')
    selected = status == 0 and '.app/Contents/Developer' in path
    checks.append(Check('Xcode selected', selected, path or 'xcode-select failed',
                        'sudo xcode-select -s /Applications/Xcode.app/Contents/Developer'))
    status, output = runner('xcodebuild', '-version')
    words = output.splitlines()[0].split() if status == 0 and output else []
    try:
        major = int(words[1].split('.')[0]) if len(words) > 1 and words[0] == 'Xcode' else 0
    except ValueError:
        major = 0
    checks.append(Check('Xcode 16 or later', major >= 16, ' '.join(words) or 'Xcode not found',
                        'Install or update Xcode from the App Store.'))
    tools = [name for name in ('metal', 'metallib') if runner('xcrun', '-f', name)[0] != 0]
    checks.append(Check('Metal compiler', not tools,
                        'missing ' + ', '.join(tools) if tools else 'metal and metallib found',
                        'xcodebuild -downloadComponent MetalToolchain'))
    if major >= 26:
        status, output = runner('xcodebuild', '-showComponent', 'MetalToolchain', '-json')
        try:
            component = json.loads(output[output.index('{'):output.rindex('}') + 1])
        except ValueError:
            component = {}
        installed = status == 0 and component.get('status') == 'installed'
        checks.append(Check('Metal toolchain component', installed,
                            f"{component.get('status', 'unknown')} {component.get('buildVersion', '')}".strip(),
                            'xcodebuild -downloadComponent MetalToolchain'))
    return checks


def mojo_check():
    try:
        return Check('Mojo in this environment', True, environment_tool('mojo'))
    except RuntimeError as error:
        return Check('Mojo in this environment', False, str(error), 'uv sync --locked')


def uv_check():
    path = shutil.which('uv')
    return Check('uv on PATH', path is not None, path or 'not found',
                 'Install uv: https://docs.astral.sh/uv/getting-started/installation/', 'prepare')


def disk_check(directory, required):
    """Free space where the store lives (its nearest existing parent)."""
    existing = directory
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    return Check('Free space for the store', free >= required,
                 f'{free / 1e9:.1f} GB free, {required / 1e9:.1f} GB needed at {directory}',
                 'Free space, or set LLM_MOJO_CACHE_DIR to another volume.', 'prepare')


def device(runner=run):
    status, output = runner('sysctl', '-n', 'machdep.cpu.brand_string')
    return output if status == 0 and output else 'unknown'


def device_check(runner=run):
    """Tuned Fast selection applies only to the measured device; others run configuration 0."""
    name = device(runner)
    if name == MEASURED_DEVICE:
        return Check('Fast kernel selection', True, f'{name}: measured configurations apply', blocks='')
    return Check('Fast kernel selection', True,
                 f'{name}: kernel choices were measured only on {MEASURED_DEVICE}; this Mac runs the '
                 'baseline configuration, which is correct but not tuned', blocks='')
