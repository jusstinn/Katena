# CASK / Jetson — connect and bootstrap

This is the playbook for getting from "I just plugged in the CASK kit" to
"the FOG neutralizer is running on the Jetson."

## 1. Find the Jetson on the network

The CASK kit usually ships pre-configured to either get an IP via DHCP on
its ethernet port, or expose itself over USB-C as a virtual ethernet
device at a fixed IP.

Try these in order:

```bash
# Option A: it's on your local network. Scan for it.
ping cask.local                     # mDNS, often works
arp -a                              # look for an NVIDIA MAC prefix

# Option B: USB-C ethernet. Check what new interface appeared.
ifconfig | grep -A1 'flags=' | grep -B1 192.168
# Common Jetson defaults: 192.168.55.1 or 192.168.7.2
```

If those fail, plug in a monitor + keyboard once to set/check the IP and
write it down for SSH access going forward.

## 2. SSH in

Default credentials on a fresh CASK kit are usually printed on the box or
in the kit docs. They almost always change for hackathon kits — ask
Palantir staff if unclear.

```bash
ssh <user>@<host>
# Drop your public key for passwordless future logins:
ssh-copy-id <user>@<host>
```

Then save the host in `~/.ssh/config` so you don't have to retype:

```
Host cask
    HostName 192.168.x.x
    User <user>
    IdentityFile ~/.ssh/id_ed25519
```

After that: `ssh cask` just works.

## 3. Transfer the project

From the MacBook, in the project root:

```bash
# One-shot copy (skip the venv to save bandwidth)
rsync -av --exclude='.venv*' --exclude='downloads' --exclude='__pycache__' \
    ~/Desktop/Katena/ cask:~/Katena/

# OR: if you've pushed to GitHub, just git clone on the Jetson.
ssh cask 'git clone <your-repo-url> ~/Katena'
```

## 4. Bootstrap the Jetson environment

```bash
ssh cask
cd ~/Katena
bash jetson/setup_jetson.sh
```

This creates `.venv-jetson/` with `--system-site-packages` so it inherits
the NVIDIA-built CUDA OpenCV and PyTorch (do NOT `pip install torch` on
Jetson — that pulls CPU-only wheels and silently breaks acceleration).

## 5. Verify

```bash
source .venv-jetson/bin/activate
python3 jetson/verify_jetson.py
```

Expected output: Jetson model name, CUDA device listed, OpenCV reports
CUDA-enabled, at least one camera index returns frames.

If a camera index opens but returns no frames AND you're using the
ribbon (CSI) camera, you need a GStreamer pipeline — see the hint
printed by `verify_jetson.py`.

## 6. Iteration loop while developing

```bash
# On MacBook, after edits:
rsync -av --exclude='.venv*' --exclude='downloads' --exclude='__pycache__' \
    ~/Desktop/Katena/ cask:~/Katena/

# On Jetson:
source ~/Katena/.venv-jetson/bin/activate
python3 ~/Katena/macbook/main_tracker.py    # (when it exists)
```

A neat shortcut is `mutagen` or `unison` for live two-way sync, but
rsync-on-save is fine for a hackathon.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ssh: connection refused` | sshd may be off — physically log in and `sudo systemctl enable --now ssh` |
| `cv2 has no attribute cuda` | OpenCV was reinstalled from pip — uninstall it and let the system one come back via `--system-site-packages` |
| `torch.cuda.is_available() == False` | You probably did `pip install torch`. Uninstall, recreate venv with `--system-site-packages` |
| Camera index opens but no frames | Try `cv2.VideoCapture(0, cv2.CAP_V4L2)` for USB cams, or a GStreamer pipeline for CSI |
| Slow YOLO inference | Confirm CUDA torch with `python3 -c "import torch; print(torch.cuda.is_available())"`; consider exporting model to TensorRT engine for 3-5x speedup |
| Network disappears | CASK has a network kill switch — that's the demo feature, not a bug. Verify the system keeps running offline. |

## CASK / OSDK specific

The CASK appliance comes with kit-specific bootstrap docs from Palantir
covering:
  - Authenticating the device against your Foundry stack
  - Configuring the Embedded Ontology projection
  - The network kill-switch toggle

Follow those docs to provision auth before running anything that hits
`foundry-platform-sdk` or the generated OSDK client.
