import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core import Document, Settings

DEFAULT_OCR_LANGS = "ita+eng"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_VERSION = 1


def configure_embedding(model_name: Optional[str] = None) -> Dict[str, str]:
    if not model_name:
        model_name = os.getenv("DOC_INDEX_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except Exception as exc:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "Missing HuggingFace embeddings. Install 'llama-index-embeddings-huggingface' "
            "and 'sentence-transformers', or set DOC_INDEX_EMBED_MODEL to a valid model."
        ) from exc
    Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)
    Settings.llm = None
    return {"type": "huggingface", "model": model_name}


def detect_platform() -> str:
    system_name = platform.system().lower()
    if system_name == "darwin":
        return "macos"
    if system_name == "linux":
        return "linux"
    if system_name == "windows":
        return "windows"
    return "unknown"


def which_cmd(name: str) -> Optional[str]:
    return shutil.which(name)


def is_tool_available(tool: str) -> bool:
    return which_cmd(tool) is not None


def run_cmd(cmd: Sequence[str], timeout: int = 1800, check: bool = True) -> Dict[str, Any]:
    completed = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    result = {
        "cmd": list(cmd),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "ok": completed.returncode == 0,
    }
    if check and completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "command failed"
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {message}")
    return result


def _command_error(cmd: Sequence[str], exc: Exception) -> Dict[str, Any]:
    return {
        "cmd": list(cmd),
        "returncode": -1,
        "stdout": "",
        "stderr": str(exc),
        "ok": False,
    }


def _looks_like_permission_error(result: Dict[str, Any]) -> bool:
    text = f"{result.get('stderr', '')}\n{result.get('stdout', '')}".lower()
    markers = (
        "permission denied",
        "operation not permitted",
        "not permitted",
        "must be root",
        "are you root",
        "requires superuser",
        "insufficient permissions",
        "could not open lock file",
        "you are not allowed",
        "a password is required",
    )
    return any(marker in text for marker in markers)


def run_with_optional_sudo(
    cmd: Sequence[str],
    allow_sudo: bool,
    platform_name: Optional[str] = None,
    timeout: int = 1800,
) -> Dict[str, Any]:
    platform_name = platform_name or detect_platform()
    attempts: List[Dict[str, Any]] = []

    try:
        first = run_cmd(cmd, timeout=timeout, check=False)
    except Exception as exc:
        first = _command_error(cmd, exc)
    attempts.append(first)
    if first.get("ok"):
        return {"ok": True, "used_sudo": False, "attempts": attempts}

    if (
        not allow_sudo
        or platform_name not in {"linux", "macos"}
        or (cmd and cmd[0] == "sudo")
        or not _looks_like_permission_error(first)
        or which_cmd("sudo") is None
    ):
        return {"ok": False, "used_sudo": False, "attempts": attempts}

    sudo_cmd = ["sudo", "-n", *list(cmd)]
    try:
        second = run_cmd(sudo_cmd, timeout=timeout, check=False)
    except Exception as exc:
        second = _command_error(sudo_cmd, exc)
    attempts.append(second)
    return {
        "ok": bool(second.get("ok")),
        "used_sudo": True,
        "attempts": attempts,
    }


def _default_tool_state(available_before: bool) -> Dict[str, Any]:
    return {
        "available_before": available_before,
        "available_after": available_before,
        "status": "available" if available_before else "missing",
        "reason": "already installed" if available_before else "not installed",
        "attempts": [],
    }


def _apply_attempts(tool_state: Dict[str, Any], cmd_result: Dict[str, Any]) -> None:
    attempts = tool_state.setdefault("attempts", [])
    for attempt in cmd_result.get("attempts", []):
        attempts.append(attempt)


def _latest_failure_reason(tool_state: Dict[str, Any]) -> str:
    attempts = tool_state.get("attempts", []) or []
    for attempt in reversed(attempts):
        if attempt.get("ok"):
            continue
        stderr = str(attempt.get("stderr") or "").strip()
        stdout = str(attempt.get("stdout") or "").strip()
        return stderr or stdout or "command failed"
    return "installation failed"


def _finalize_tool_state(
    tool_name: str,
    tool_state: Dict[str, Any],
    platform_name: str,
    unsupported: bool = False,
) -> None:
    tool_state["available_after"] = is_tool_available(tool_name)
    if tool_state["available_after"]:
        if tool_state["available_before"]:
            tool_state["status"] = "available"
            tool_state["reason"] = "already installed"
        else:
            tool_state["status"] = "installed"
            tool_state["reason"] = "installed successfully"
        return

    if unsupported:
        tool_state["status"] = "unsupported"
        tool_state["reason"] = "auto-install not supported on this platform"
        return

    tool_state["status"] = "failed"
    if tool_name == "unpaper" and platform_name == "windows":
        tool_state["reason"] = "native Windows auto-install is unsupported"
    else:
        tool_state["reason"] = _latest_failure_reason(tool_state)


def install_ocr_tools(
    platform_name: Optional[str] = None,
    allow_sudo: bool = True,
    timeout: int = 3600,
) -> Dict[str, Any]:
    platform_name = platform_name or detect_platform()
    tool_states: Dict[str, Dict[str, Any]] = {
        "ocrmypdf": _default_tool_state(is_tool_available("ocrmypdf")),
        "unpaper": _default_tool_state(is_tool_available("unpaper")),
    }

    result: Dict[str, Any] = {
        "platform": platform_name,
        "attempted": False,
        "package_manager": None,
        "tools": tool_states,
        "success": False,
        "full_pipeline_available": False,
    }

    missing_tools = [name for name, state in tool_states.items() if not state["available_before"]]
    if not missing_tools:
        result["success"] = True
        result["full_pipeline_available"] = True
        return result

    result["attempted"] = True

    if platform_name == "macos":
        if not which_cmd("brew"):
            for tool in missing_tools:
                tool_states[tool]["status"] = "failed"
                tool_states[tool]["reason"] = "Homebrew not found"
        else:
            result["package_manager"] = "brew"
            for tool in missing_tools:
                command_result = run_with_optional_sudo(
                    ["brew", "install", tool],
                    allow_sudo=allow_sudo,
                    platform_name=platform_name,
                    timeout=timeout,
                )
                _apply_attempts(tool_states[tool], command_result)

    elif platform_name == "linux":
        install_targets = [tool for tool in missing_tools if tool in {"ocrmypdf", "unpaper"}]
        if not install_targets:
            pass
        elif which_cmd("apt-get"):
            result["package_manager"] = "apt-get"
            update_result = run_with_optional_sudo(
                ["apt-get", "update"],
                allow_sudo=allow_sudo,
                platform_name=platform_name,
                timeout=timeout,
            )
            for tool in install_targets:
                _apply_attempts(tool_states[tool], update_result)
            install_result = run_with_optional_sudo(
                ["apt-get", "install", "-y", *install_targets],
                allow_sudo=allow_sudo,
                platform_name=platform_name,
                timeout=timeout,
            )
            for tool in install_targets:
                _apply_attempts(tool_states[tool], install_result)
        elif which_cmd("dnf"):
            result["package_manager"] = "dnf"
            install_result = run_with_optional_sudo(
                ["dnf", "install", "-y", *install_targets],
                allow_sudo=allow_sudo,
                platform_name=platform_name,
                timeout=timeout,
            )
            for tool in install_targets:
                _apply_attempts(tool_states[tool], install_result)
        elif which_cmd("pacman"):
            result["package_manager"] = "pacman"
            install_result = run_with_optional_sudo(
                ["pacman", "-Sy", "--noconfirm", *install_targets],
                allow_sudo=allow_sudo,
                platform_name=platform_name,
                timeout=timeout,
            )
            for tool in install_targets:
                _apply_attempts(tool_states[tool], install_result)
        else:
            for tool in install_targets:
                tool_states[tool]["status"] = "failed"
                tool_states[tool]["reason"] = "No supported Linux package manager found"

    elif platform_name == "windows":
        result["package_manager"] = "winget+pip"
        if not tool_states["ocrmypdf"]["available_before"]:
            if which_cmd("winget"):
                winget_flags = ["--accept-source-agreements", "--accept-package-agreements"]
                for package_id in (
                    "Ghostscript.Ghostscript",
                    "TesseractOCR.Tesseract",
                ):
                    winget_result = run_with_optional_sudo(
                        ["winget", "install", "--id", package_id, *winget_flags],
                        allow_sudo=False,
                        platform_name=platform_name,
                        timeout=timeout,
                    )
                    _apply_attempts(tool_states["ocrmypdf"], winget_result)
            else:
                tool_states["ocrmypdf"]["status"] = "failed"
                tool_states["ocrmypdf"]["reason"] = "winget not found"

            py_command: Optional[List[str]] = None
            if which_cmd("py"):
                py_command = ["py", "-m", "pip", "install", "--upgrade", "ocrmypdf"]
            elif which_cmd("python"):
                py_command = ["python", "-m", "pip", "install", "--upgrade", "ocrmypdf"]
            elif which_cmd("python3"):
                py_command = ["python3", "-m", "pip", "install", "--upgrade", "ocrmypdf"]

            if py_command:
                pip_result = run_with_optional_sudo(
                    py_command,
                    allow_sudo=False,
                    platform_name=platform_name,
                    timeout=timeout,
                )
                _apply_attempts(tool_states["ocrmypdf"], pip_result)
            else:
                tool_states["ocrmypdf"]["status"] = "failed"
                tool_states["ocrmypdf"]["reason"] = "Python launcher not found"

        if not tool_states["unpaper"]["available_before"]:
            tool_states["unpaper"]["status"] = "unsupported"
            tool_states["unpaper"]["reason"] = "native Windows auto-install is unsupported"

    else:
        for tool in missing_tools:
            tool_states[tool]["status"] = "failed"
            tool_states[tool]["reason"] = "Unsupported platform"

    _finalize_tool_state("ocrmypdf", tool_states["ocrmypdf"], platform_name)
    _finalize_tool_state(
        "unpaper",
        tool_states["unpaper"],
        platform_name,
        unsupported=(platform_name == "windows" and not is_tool_available("unpaper")),
    )

    result["full_pipeline_available"] = bool(
        tool_states["ocrmypdf"]["available_after"] and tool_states["unpaper"]["available_after"]
    )
    required = ["ocrmypdf"] if platform_name == "windows" else ["ocrmypdf", "unpaper"]
    result["success"] = all(tool_states[name]["available_after"] for name in required)
    return result


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_binary_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(2048)
    except Exception:
        return False
    if not chunk:
        return False
    if b"\0" in chunk:
        return True
    nontext = 0
    for byte in chunk:
        if byte in (9, 10, 13) or 32 <= byte <= 126:
            continue
        nontext += 1
    return (nontext / len(chunk)) > 0.3


def save_docs_cache(
    path: Path, docs: List[Document], file_path: str, file_type: str
) -> None:
    payload: Dict[str, Any] = {
        "version": CACHE_VERSION,
        "file_path": file_path,
        "file_type": file_type,
        "docs": [{"text": doc.text, "metadata": doc.metadata} for doc in docs],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))


def load_docs_cache(path: Path) -> List[Document]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    docs = []
    for item in payload.get("docs", []):
        docs.append(Document(text=item.get("text", ""), metadata=item.get("metadata") or {}))
    return docs
