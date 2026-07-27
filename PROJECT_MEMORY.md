---
name: selfdriving-unified
description: "Self-driving campus cart / RC car — hardware stack, AI command architecture (YOLO+Gemini+state machine over DonkeyCar), GitHub access notes"
metadata:
  node_type: memory
  type: project
---

Self-driving RC car project. GitHub repo: **https://github.com/GuptaOum/selfdriving-campus-cart** (account **GuptaOum**).

---

## Hardware architecture

```
FlySky TX → Receiver → Raspberry Pi → PCA9685 → Servo + ESC
```

- Receiver sends control signals via iBUS/PPM (preferred over raw PWM).
- Raspberry Pi reads + processes signals (acts as middle controller/translator).
- PCA9685 outputs stable jitter-free PWM to servo and ESC.
- All components share common GND; servo/ESC use external power.

**Why:** PCA9685 gives better, jitter-free PWM compared to driving servos directly from GPIO — works well with the DonkeyCar framework.

**How to apply:** when suggesting wiring, code, or config changes, assume this hardware chain. Default to iBUS/PPM for the receiver protocol. An external power rail for servo/ESC is required.

---

## AI command architecture plan

Split cleanly between PC (all AI/vision compute) and Pi (drives + reacts):

```
Pi camera → frames → PC
PC: YOLO builds a "track memory" (detected objects + timestamps/positions)
User types a natural-language prompt on PC
PC: Gemini receives {prompt + track memory (text only, no images)} → structured JSON command
PC → Pi: command sent over WiFi socket
Pi: state machine — DonkeyCar drives normally, YOLO watches live;
    when target detected AND bounding-box area exceeds threshold → execute on_reach action (stop, do_360, etc.)
```

**Why this split:** PC handles all AI/vision compute; Pi just drives and reacts. The LLM (Gemini) only does text reasoning over the track-memory summary, never touches images directly — keeps the Pi-side loop simple and fast.

### Command structure
```json
{"action": "go_to", "target": "parking", "on_reach": "stop"}
{"action": "go_to", "target": "stop_sign", "on_reach": "do_360"}
```

### Build phases
1. Stop/resume override in `manage.py`.
2. YOLO on PC + track memory + "go to target".
3. Gemini command parsing.
4. Smart behaviors (alignment, indexing, sub-states).

**How to apply:** keep PC-side and Pi-side code clearly separated when implementing. PC = YOLO + Gemini + track memory + socket sender. Pi = DonkeyCar model + socket receiver + state machine + override controller in `manage.py`.

---

## GitHub access notes

- Git pushes authenticate via **Git Credential Manager** (`credential.helper=manager`); its OAuth token also works for API calls including repo creation — retrieve with `git credential fill`.
- A fine-grained PAT stored elsewhere (`E:\proj\SchlorRag\ScholarRAG\backend\.env`, `GITHUB_TOKEN`) can read/push existing repos but **cannot create repos** ("Resource not accessible").
- `gh` CLI is installed (`C:\Program Files\GitHub CLI\gh.exe`) but not logged in; it also hit TLS handshake timeouts while plain `curl` worked.

**Why:** avoids re-discovering the auth path each session.
**How to apply:** for GitHub API tasks on this machine, prefer the GCM credential via `git credential fill`; never print the token.
