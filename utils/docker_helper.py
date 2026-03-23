"""
utils/docker_helper.py
Fixes:
  1. LLM drops --extra-index-url  → injected programmatically, never trusted to LLM
  2. libgl1 pulls 217MB Mesa/LLVM  → removed (not needed for headless PyTorch CPU)
  3. Dockerfile template shown to LLM now matches the lean fallback exactly
  4. Container run: no network_mode=host, no privileged, no remove=True race condition
  5. Robust fenced-block parsing handles ```python / ```dockerfile language tags
  6. CODE_TIMEOUT_SEC and MAX_MEMORY_MB now read from config
  7. Data folder (MovieLens, etc.) copied into Docker build context if present
"""

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import re
import docker
import tempfile
from config import GROQ_MODEL, GROQ_TEMPERATURE, CODE_TIMEOUT_SEC, MAX_MEMORY_MB

load_dotenv()

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=GROQ_TEMPERATURE,
    max_tokens=2048,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PYTORCH_CPU_INDEX = "--extra-index-url https://download.pytorch.org/whl/cpu"
_APT_PACKAGES = "libgomp1 libjpeg62-turbo libpng16-16"

FALLBACK_REQUIREMENTS = f"""\
{_PYTORCH_CPU_INDEX}
torch==2.3.0+cpu
torchvision==0.18.0+cpu
numpy
pillow
"""

FALLBACK_DOCKERFILE = f"""\
FROM python:3.10-slim-bookworm

RUN apt-get update -y && \\
    apt-get install -y --no-install-recommends {_APT_PACKAGES} && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/data && chown -R nobody:nogroup /app/data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER nobody

CMD ["python", "solution.py"]
"""

_LLM_DOCKERFILE_TEMPLATE = f"""\
FROM python:3.10-slim-bookworm

RUN apt-get update -y && \\
    apt-get install -y --no-install-recommends {_APT_PACKAGES} && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/data && chown -R nobody:nogroup /app/data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER nobody

CMD ["python", "solution.py"]"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_code_blocks(text: str) -> list[str]:
    pattern = re.compile(r"```[a-zA-Z0-9._-]*\n?(.*?)```", re.DOTALL)
    return [b.strip() for b in pattern.findall(text) if b.strip()]


def _clean_requirements(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("**", "*", "#")) or "requirements" in stripped.lower():
            continue
        if "--extra-index-url" in stripped or "--index-url" in stripped:
            continue
        lines.append(stripped)

    packages = "\n".join(lines)
    return f"{_PYTORCH_CPU_INDEX}\n{packages}"


def _clean_dockerfile(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("**", "*", "#")) or stripped.lower() == "dockerfile":
            continue
        lines.append(stripped)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_docker_files(code: str, paper_id: str = "temp") -> dict:
    print(f"🐳 Generating Docker files for {paper_id}...")

    prompt = f"""You are a precise DevOps expert. Output EXACTLY two fenced code blocks — nothing else.
No introductions, no explanations, no text before or after the blocks.

Given this Python code:
{code}

Block 1 — requirements.txt (add only packages actually imported in the code above):
```
torch==2.3.0+cpu
torchvision==0.18.0+cpu
numpy
pillow
```

Block 2 — Dockerfile (copy exactly, only change CMD if the script name differs):
```
{_LLM_DOCKERFILE_TEMPLATE}
```

Important rules:
- Do NOT include --extra-index-url in requirements.txt (it will be added automatically).
- Do NOT change the base image or apt packages in the Dockerfile.
- Do NOT add any text outside the two code blocks."""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        print("Raw Groq response:\n", content)

        blocks = _extract_code_blocks(content)

        if len(blocks) < 2:
            raise ValueError(f"Expected 2 code blocks, got {len(blocks)}")

        reqs = _clean_requirements(blocks[0])
        dockerfile = _clean_dockerfile(blocks[1])

        if not reqs:
            raise ValueError("requirements.txt is empty after cleaning")
        if "FROM" not in dockerfile:
            raise ValueError("Dockerfile missing FROM instruction")

        print("   ✅ Parsed successfully")
        print("   requirements.txt preview:\n", reqs[:300])
        return {"requirements": reqs, "dockerfile": dockerfile}

    except Exception as e:
        print(f"⚠️  LLM parse failed ({e}) — using fallback files")
        return {
            "requirements": FALLBACK_REQUIREMENTS,
            "dockerfile": FALLBACK_DOCKERFILE,
        }


def run_code_in_docker(
    code: str,
    paper_id: str,
    timeout: int = CODE_TIMEOUT_SEC,
) -> tuple[bool, str, str]:
    """
    Build a Docker image and run solution.py inside it.
    Reuses existing image if already built — skips Groq call entirely.

    Returns: (success, logs, error_msg)
    """
    print(f"🐳 Starting Docker execution for {paper_id}...")

    client = docker.from_env()
    image_tag = f"repro-{re.sub(r'[^a-z0-9-]', '-', paper_id.lower())}"

    with tempfile.TemporaryDirectory() as tmpdir:

        # Always write solution.py — needed for container run
        with open(os.path.join(tmpdir, "solution.py"), "w", encoding="utf-8") as f:
            f.write(code)

        # ── Resolve local data folder path (used for volume mount) ─────────
        local_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        local_data_dir = os.path.normpath(local_data_dir)
        os.makedirs(local_data_dir, exist_ok=True)
        print(f"   📂 Will mount data folder as volume: {local_data_dir}")

        # ── Check if image already exists — skip Groq + build if so ───────
        image = None
        try:
            image = client.images.get(image_tag)
            print(f"   ♻️  Reusing existing image: {image_tag} (skip rebuild)")
        except docker.errors.ImageNotFound:
            pass

        # ── Build only when image not found ────────────────────────────────
        if image is None:
            docker_files = generate_docker_files(code, paper_id)

            with open(os.path.join(tmpdir, "requirements.txt"), "w", encoding="utf-8") as f:
                f.write(docker_files["requirements"])
            with open(os.path.join(tmpdir, "Dockerfile"), "w", encoding="utf-8") as f:
                f.write(docker_files["dockerfile"])

            print("\n   📄 requirements.txt to be used:")
            print(docker_files["requirements"])
            print("\n   📄 Dockerfile to be used:")
            print(docker_files["dockerfile"])
            print("\n   Building Docker image (5-15 min first time)...")

            try:
                image, _ = client.images.build(
                    path=tmpdir,
                    tag=image_tag,
                    rm=True,
                    forcerm=True,
                    nocache=False,
                    quiet=False,
                )
                print("   ✅ Image built successfully")

            except docker.errors.BuildError as e:
                build_output = "\n".join(
                    line.get("stream", line.get("error", ""))
                    for line in e.build_log
                    if isinstance(line, dict)
                ).strip()
                msg = f"Build failed:\n{build_output}"
                print(f"❌ {msg}")
                return False, "", msg

            except Exception as e:
                msg = f"Build failed (unexpected): {e}"
                print(f"❌ {msg}")
                return False, "", msg

        # ── Run container ──────────────────────────────────────────────────
        print("   Starting container...")
        container = None
        try:
            container = client.containers.run(
                image.id,
                detach=True,
                mem_limit=f"{MAX_MEMORY_MB}m",
                volumes={local_data_dir: {'bind': '/app/data', 'mode': 'rw'}},
                stdout=True,
                stderr=True,
            )

            logs = ""
            for chunk in container.logs(stream=True, follow=True):
                decoded = chunk.decode(errors="ignore")
                logs += decoded
                print(decoded, end="", flush=True)

            result = container.wait(timeout=timeout)
            exit_code = result["StatusCode"]
            success = exit_code == 0

            print(f"\n   Container finished. Exit code: {exit_code}. Success: {success}")
            return success, logs, ("" if success else f"Exit code {exit_code}\n{logs}")

        except Exception as e:
            msg = f"Container error: {e}"
            print(f"❌ {msg}")
            return False, "", msg

        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                    print("   🧹 Container removed")
                except Exception:
                    pass