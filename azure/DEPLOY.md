# Lattice Pulse on Azure (Microsoft for Startups credits)

Same API as Modal. Vercel keeps using `MODAL_API_URL` + `LATTICE_API_SECRET`.

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
cd ~/azure   # or wherever you put server.py
pip install -r requirements.txt
# torch CUDA wheel if pip CPU torch got installed:
# pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

export LATTICE_API_SECRET='same-secret-as-vercel'
export MODEL_ID='lattice-research/lattice-pulse'
# if private model: export HF_TOKEN=hf_...

uvicorn server:web --host 0.0.0.0 --port 8000
```

Test:

```bash
curl -s http://YOUR_PUBLIC_IP:8000/health
curl -s -X POST http://YOUR_PUBLIC_IP:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Lattice-Secret: $LATTICE_API_SECRET" \
  -d '{"message":"who are you?","history":[]}'
```

## Point Vercel at Azure

Vercel env:
- `MODAL_API_URL` = `http://YOUR_PUBLIC_IP:8000`  
  (for production use HTTPS; for a private demo HTTP works if you accept the risk)
- `LATTICE_API_SECRET` = same value as on the VM

Redeploy / wait for env to apply → open `pulse.html`.

## Dual model (Pulse 1 + Pulse 2)

The site model picker sends `model: "pulse" | "pulse2"`.

On the VM:

```bash
# Install extra deps once
cd ~/pulse && source venv/bin/activate
pip install -U peft bitsandbytes

# Copy Pulse 2.0 checkpoint-400 from your Mac (adapter only ~175MB)
mkdir -p ~/pulse/adapters/pulse2-checkpoint-400
# from Mac:
# scp -i KEY -r ".../results/lattice-pulse-2-8b-lora/checkpoint-400/"* \
#   azureuser@IP:~/pulse/adapters/pulse2-checkpoint-400/

export PULSE1_MODEL_ID=lattice-research/lattice-pulse
export PULSE2_BASE=Qwen/Qwen3-8B
export PULSE2_ADAPTER=$HOME/pulse/adapters/pulse2-checkpoint-400
export LATTICE_API_SECRET="$(cat .secret)"

# Copy latest server.py from lattice-site/azure/server.py then:
pkill -f "uvicorn server:web" || true
nohup uvicorn server:web --host 0.0.0.0 --port 8000 > ~/pulse/server.log 2>&1 &
```

First Pulse 2 request downloads Qwen3-8B 4-bit (~several GB) — needs free disk.

## Cost tip

Deallocate from Portal → VM → **Stop** when idle. First message after start will be slow (model reload).
