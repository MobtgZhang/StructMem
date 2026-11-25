from __future__ import annotations

import os
import selectors
import subprocess
import threading
from pathlib import Path
from typing import Any


def patch_textworld_eval() -> None:
    """Make TextWorld expression evaluation work on Python 3.13+."""
    from textworld.envs.pddl.textgen import EvalSymbol, TerminalSymbol

    if getattr(EvalSymbol, "_structmem_explicit_eval_namespace", False):
        return

    def derive(self: Any, context: dict[str, Any] | None = None) -> list[Any]:
        active = context or self.context
        variables = dict(active.get("variables", {}))
        value = eval(self.expression, {"__builtins__": {}}, variables)
        return [TerminalSymbol(value)]

    EvalSymbol.derive = derive
    EvalSymbol._structmem_explicit_eval_namespace = True


def patch_scienceworld_partial_close(scienceworld_module: Any) -> None:
    """Avoid destructor noise when JVM startup fails before _gateway exists."""
    cls = scienceworld_module.ScienceWorldEnv
    if getattr(cls, "_structmem_safe_partial_close", False):
        return
    original_close = cls.close

    def close(self: Any) -> None:
        if hasattr(self, "_gateway"):
            original_close(self)

    cls.close = close
    cls.__del__ = close
    cls._structmem_safe_partial_close = True


def bounded_py4j_launch_gateway(
    port: int = 0,
    jarpath: str = "",
    classpath: str = "",
    javaopts: list[str] | None = None,
    die_on_exit: bool = False,
    redirect_stdout: Any = None,
    redirect_stderr: Any = None,
    daemonize_redirect: bool = True,
    java_path: str = "java",
    create_new_process_group: bool = False,
    enable_auth: bool = False,
    cwd: str | None = None,
    return_proc: bool = False,
    use_shell: bool = False,
    *,
    startup_timeout: float = 30.0,
) -> Any:
    """Launch Py4J without its unbounded ``stdout.readline()`` startup wait."""
    from py4j.java_gateway import find_jar_path

    if enable_auth:
        raise RuntimeError("The bounded ScienceWorld launcher does not support Py4J authentication")
    py4j_jar = jarpath or find_jar_path()
    if not Path(py4j_jar).exists():
        raise RuntimeError(f"Py4J jar not found: {py4j_jar}")
    java = java_path or str(Path(os.environ.get("JAVA_HOME", "")) / "bin" / "java")
    command = [java, "-classpath", os.pathsep.join((py4j_jar, classpath)), *(javaopts or []), "py4j.GatewayServer"]
    if die_on_exit:
        command.append("--die-on-broken-pipe")
    command.append(str(port))
    creationflags = 0
    start_new_session = bool(create_new_process_group)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        shell=use_shell,
        start_new_session=start_new_session,
        creationflags=creationflags,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    events = selector.select(startup_timeout)
    if not events:
        _stop_process(process)
        detail = process.stderr.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Py4J gateway did not publish a port within {startup_timeout:g}s"
            + (f": {detail[-1000:]}" if detail else "")
        )
    line = process.stdout.readline().decode("utf-8", errors="replace").strip()
    if not line:
        _stop_process(process)
        detail = process.stderr.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Py4J gateway exited before publishing a port: {detail[-1000:]}")
    try:
        selected_port = int(line)
    except ValueError as exc:
        _stop_process(process)
        detail = process.stderr.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Invalid Py4J gateway port {line!r}: {detail[-1000:]}") from exc

    _drain_stream(process.stderr, redirect_stderr, daemonize_redirect)
    _drain_stream(process.stdout, redirect_stdout, daemonize_redirect)
    return (selected_port, process) if return_proc else selected_port


def _drain_stream(stream: Any, target: Any, daemon: bool) -> None:
    def consume() -> None:
        for raw in iter(stream.readline, b""):
            if target is None:
                continue
            line = raw.decode("utf-8", errors="replace")
            if hasattr(target, "put"):
                target.put(line)
            elif hasattr(target, "appendleft"):
                target.appendleft(line)
            elif hasattr(target, "write"):
                target.write(line)

    threading.Thread(target=consume, daemon=daemon).start()


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
