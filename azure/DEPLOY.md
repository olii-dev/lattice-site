# Lattice Quark + Spark on Azure (Microsoft for Startups credits)

Same API as before. Vercel keeps using `MODAL_API_URL` + `LATTICE_API_SECRET`.

Models (resident on the T4, no hot-swap):
- `quark` — 1.5B from-scratch nanochat model + custom tokenizer
- `spark` — 1.5B Qwen2.5 identity LoRA (transformers)

## Billing — keep the card at $0

1. **Stay on credits.** Do **not** click “upgrade to pay-as-you-go” / remove spending protection.
2. **Portal → Cost Management + Billing → Budgets**  
   Create a budget for this subscription:
   - Amount: `$50` (alert) and another at `$900`
   - Alert emails: your email
3. **Portal → Subscriptions → your sub → Cost alerts** if available — turn on.
4. **When not demoing:** stop the VM (**Stop / Deallocate**).  
   Stopped-but-allocated still bills GPU. **Deallocate** = almost no compute charge (disk may still cost a little).
5. Done for the week? **Delete the resource group** so nothing keeps billing.
6. Watch the credit balance on the Startups dashboard (`$1,000 … Exp Oct 20, 2026`).

Credits burn first. Card should not charge while spending protection is on and you have credit left. The failure mode is: credits hit $0 **and** you flipped to PAYG with a VM still running.

## Create the GPU VM (portal clicks)

1. Azure Portal → **Create a resource** → **Virtual machine**
2. Basics:
   - Resource group: `lattice-pulse` (new)
   - Name: `pulse-gpu`
   - Region: try **Australia East**, else **East US**, **West Europe** (GPU stock varies)
   - Image: **Ubuntu 22.04 LTS**
   - Size: click **See all sizes** → search **T4** or **NC** / **NCasT4_v3**  
     Prefer **Standard_NC4as_T4_v3** (1× T4) if listed. If none, try another region.
   - Auth: SSH public key (create new or paste yours)
3. Disks: default OS disk (Premium SSD is fine; cheaper Standard SSD OK for demo)
4. Networking:
   - Public IP: yes
   - NIC NSG: allow **SSH (22)** and **HTTP 8000** (or 80 if you terminate TLS later)
5. Review + create → wait ~5 min → note the **public IP**

## Install CUDA + server (SSH)

```bash
ssh azureuser@YOUR_PUBLIC_IP

# NVIDIA drivers (Ubuntu 22.04)
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
# ssh back in, then:
nvidia-smi   # should show Tesla T4 (or similar)

sudo apt-get install -y python3-pip python3-venv git
python3 -m venv ~/pulse-venv
source ~/pulse-venv/bin/activate
pip install -U pip

# copy azure/ from your laptop, or:
# scp -r azure azureuser@IP:~/
cd ~/pulse   # server.py lives here on the VM
pip install -r requirements.txt
# torch CUDA wheel if pip CPU torch got installed:
# pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# The VM's start script (~/pulse/start.sh) reads the secret from ~/pulse/.secret
# and runs uvicorn on port 8000:
echo 'your-secret' > ~/pulse/.secret   # keep in sync with Vercel LATTICE_API_SECRET
nohup ~/pulse/start.sh > ~/pulse/server.log 2>&1 &
```

Prereqs on the VM for the quark model (once):
- `pip install rustbpe tiktoken`
- nanochat checkout at `~/nanochat` (the repo's model code)
- `~/.cache/nanochat/tokenizer/` — tokenizer.pkl + token_bytes.pt
- `~/.cache/nanochat/chatsft_checkpoints/quark-1.5b/` — model_000465.pt + meta_000465.json
  (fetch both from `lattice-research/lattice-quark-1.5b`; convert any bf16 tensors to
  fp32 — the T4 can't run bf16)
- `pip install` deps, then `/warm {"model":"all"}` loads quark + spark once

Test:

```bash
curl -s http://YOUR_PUBLIC_IP:8000/health
curl -s -X POST http://YOUR_PUBLIC_IP:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Lattice-Secret: $(cat ~/pulse/.secret)" \
  -d '{"message":"who are you?","history":[],"model":"quark"}'
```

## Point Vercel at Azure

Vercel env:
- `MODAL_API_URL` = `http://YOUR_PUBLIC_IP:8000`  
  (for production use HTTPS; for a private demo HTTP works if you accept the risk)
- `LATTICE_API_SECRET` = same value as on the VM

Redeploy / wait for env to apply → open `chat/`.

## Retired

Pulse 1 and Pulse 2 are retired. The current server (`azure/server.py`) serves
`quark` + `spark` only. To bring a model back, wire it into `load_quark` /
`load_spark` and add it to the site's `api/chat.js` `ALLOWED` set and the
picker in `chat/index.html`.

## Cost tip

Deallocate from Portal → VM → **Stop** when idle. First message after start will be slow (model reload).
