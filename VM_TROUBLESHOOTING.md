# VM Server Troubleshooting Guide

## Problem
Clients cannot connect to Flower server running on VM at `140.245.224.116:8080`

**Error:** `StatusCode.UNAVAILABLE - failed to connect to all addresses; tcp handshaker shutdown`

## Root Cause
Port 8080 is blocked by firewall on the VM.

---

## Step-by-Step Fix

### 1. Connect to Your VM

Replace `/path/to/your-key.pem` with the actual path to your SSH private key:

```bash
# Connect to VM with your private key
ssh -i /path/to/your-key.pem ubuntu@140.245.224.116

# Example if your key is in ~/.ssh/
# ssh -i ~/.ssh/my-vm-key.pem ubuntu@140.245.224.116
```

**Note:** If you get a "permissions too open" error:
```bash
chmod 400 /path/to/your-key.pem
```

---

### 2. Check if Server is Running (on VM)

Once connected to the VM:

```bash
# Check if server process is running
ps aux | grep run_server_only

# OR check any Python server process
ps aux | grep python | grep -i server
```

**Expected:** You should see a Python process running `run_server_only.py`

---

### 3. Check if Port 8080 is Listening (on VM)

```bash
# Check which ports are listening
sudo netstat -tulpn | grep 8080

# OR use ss command
sudo ss -tulpn | grep 8080

# OR use lsof
sudo lsof -i :8080
```

**Expected output:**
```
tcp   0   0  0.0.0.0:8080   0.0.0.0:*   LISTEN
```

**Bad output (listening only locally):**
```
tcp   0   0  127.0.0.1:8080   0.0.0.0:*   LISTEN
```

If port is listening on `127.0.0.1`, the server needs to be restarted with `--host 0.0.0.0`

---

### 4. Open Firewall Port 8080 (on VM)

#### For Ubuntu/Debian (UFW):
```bash
# Allow port 8080
sudo ufw allow 8080/tcp

# Check firewall status
sudo ufw status

# If UFW is inactive, enable it
sudo ufw enable
```

#### For RedHat/CentOS/Fedora (firewalld):
```bash
# Allow port 8080
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload

# Check firewall status
sudo firewall-cmd --list-all
```

#### For iptables:
```bash
# Allow port 8080
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT

# Save rules (Ubuntu/Debian)
sudo netfilter-persistent save

# OR (RedHat/CentOS)
sudo service iptables save
```

---

### 5. Configure Cloud Provider Security Groups

If your VM is on AWS, Azure, or GCP, you also need to open port 8080 in the cloud console:

#### AWS (Security Groups):
1. Go to EC2 Console → Security Groups
2. Select the security group attached to your VM
3. Add Inbound Rule:
   - Type: Custom TCP
   - Port: 8080
   - Source: `0.0.0.0/0` (or `10.23.0.195/32` for just your PC)

#### Azure (Network Security Groups):
1. Go to Virtual Machines → Your VM → Networking
2. Add inbound port rule:
   - Destination port: 8080
   - Protocol: TCP
   - Source: `*` (or `10.23.0.195` for just your PC)

#### GCP (Firewall Rules):
1. Go to VPC Network → Firewall Rules
2. Create firewall rule:
   - Targets: All instances (or specific tag)
   - Source IP ranges: `0.0.0.0/0` (or `10.23.0.195/32`)
   - Protocols and ports: tcp:8080

---

### 6. Start Server Correctly (on VM)

If server is not running or listening on wrong interface:

```bash
# Navigate to project directory
cd /path/to/FL_CognitiveDefence

# Activate virtual environment if needed
# source venv/bin/activate

# Start server listening on all interfaces
python run_server_only.py \
  --config experiments/configs/baseline_experiment.yaml \
  --host 0.0.0.0 \
  --port 8080
```

**Important:** The `--host 0.0.0.0` flag makes the server listen on all network interfaces, not just localhost.

Keep this terminal open - the server will run in the foreground.

---

### 7. Test Connection from Local PC

Open a **new terminal on your local PC** (not on the VM):

```bash
# Test basic connectivity
ping 140.245.224.116

# Test if port 8080 is open
nc -zv 140.245.224.116 8080

# OR use telnet
telnet 140.245.224.116 8080

# OR use the diagnostic script
cd /Users/hanafemira/development/FL_CognitiveDefence
./scripts/test_connection.sh 140.245.224.116 8080
```

**Expected result:**
```
✅ Port 8080 is OPEN and accepting connections
```

---

### 8. Run Your Experiment

Once port 8080 is accessible:

```bash
# On your local PC
cd /Users/hanafemira/development/FL_CognitiveDefence
make run-baseline
```

---

## Quick Reference Commands

### Connect to VM:
```bash
ssh -i /path/to/your-key.pem ubuntu@140.245.224.116
```

### Check Server Status (on VM):
```bash
ps aux | grep run_server_only
sudo netstat -tulpn | grep 8080
```

### Open Firewall (on VM):
```bash
sudo ufw allow 8080/tcp
sudo ufw status
```

### Start Server (on VM):
```bash
python run_server_only.py \
  --config experiments/configs/baseline_experiment.yaml \
  --host 0.0.0.0 --port 8080
```

### Test Connection (on local PC):
```bash
nc -zv 140.245.224.116 8080
```

---

## Common Issues

### Issue: "Permission denied (publickey)"
- Your SSH key is not recognized
- **Fix:** Make sure you're using the correct key file and username
  ```bash
  ssh -i /path/to/correct-key.pem ubuntu@140.245.224.116
  # or try: ssh -i /path/to/correct-key.pem ec2-user@140.245.224.116
  ```

### Issue: "WARNING: UNPROTECTED PRIVATE KEY FILE!"
- SSH key has wrong permissions
- **Fix:** 
  ```bash
  chmod 400 /path/to/your-key.pem
  ```

### Issue: Server runs but clients can't connect
- Firewall is blocking port 8080
- **Fix:** Follow steps 4 and 5 above

### Issue: Connection refused immediately
- Server is not running or not listening on correct port
- **Fix:** Follow step 6 above

### Issue: Connection times out after 20 seconds
- Firewall/Security Group is blocking the port
- **Fix:** Follow steps 4 and 5 above

---

## Verification Checklist

Before running `make run-baseline`, verify:

- [ ] Can SSH into VM with private key
- [ ] Server is running on VM (`ps aux | grep run_server_only`)
- [ ] Port 8080 is listening on `0.0.0.0:8080` (not `127.0.0.1:8080`)
- [ ] VM firewall allows port 8080 (`sudo ufw status`)
- [ ] Cloud Security Group/NSG allows port 8080
- [ ] Can connect to port 8080 from local PC (`nc -zv 140.245.224.116 8080`)
