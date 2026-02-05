#!/bin/bash

# Connection diagnostic script for Flower FL server
# Usage: ./scripts/test_connection.sh <server_ip> <port>

SERVER_IP=${1:-"140.245.224.116"}
PORT=${2:-"8080"}

echo "=========================================="
echo "Flower FL Server Connection Diagnostics"
echo "=========================================="
echo "Server: ${SERVER_IP}:${PORT}"
echo ""

# Test 1: Ping test
echo "1. Testing basic connectivity (ping)..."
if ping -c 3 -W 2 ${SERVER_IP} > /dev/null 2>&1; then
    echo "   ✅ Server is reachable via ping"
else
    echo "   ❌ Server is NOT reachable via ping"
    echo "   This could indicate network issues or ICMP blocked"
fi
echo ""

# Test 2: Port connectivity with netcat
echo "2. Testing port ${PORT} connectivity..."
if command -v nc > /dev/null 2>&1; then
    if nc -z -v -w 5 ${SERVER_IP} ${PORT} 2>&1 | grep -q "succeeded\|open"; then
        echo "   ✅ Port ${PORT} is OPEN and accepting connections"
    else
        echo "   ❌ Port ${PORT} is CLOSED or FILTERED (not accessible)"
        echo "   Possible causes:"
        echo "      - Firewall blocking port ${PORT}"
        echo "      - Server not running"
        echo "      - Server listening on wrong interface (127.0.0.1 instead of 0.0.0.0)"
    fi
else
    echo "   ⚠️  netcat (nc) not found, skipping port test"
fi
echo ""

# Test 3: Telnet test
echo "3. Testing with telnet..."
if command -v telnet > /dev/null 2>&1; then
    timeout 5 telnet ${SERVER_IP} ${PORT} 2>&1 | head -n 5
    echo "   If you see 'Connected to...', the port is open"
    echo "   If you see 'Connection refused', server is not listening"
    echo "   If it hangs/timeouts, firewall is likely blocking"
else
    echo "   ⚠️  telnet not found, skipping test"
fi
echo ""

# Test 4: Check local network info
echo "4. Local network information..."
echo "   Your local IP addresses:"
ifconfig 2>/dev/null | grep "inet " | grep -v 127.0.0.1 | awk '{print "   - " $2}' || \
    ip addr 2>/dev/null | grep "inet " | grep -v 127.0.0.1 | awk '{print "   - " $2}'
echo ""

# Test 5: DNS resolution
echo "5. Testing DNS resolution for ${SERVER_IP}..."
if host ${SERVER_IP} > /dev/null 2>&1; then
    echo "   ✅ DNS resolves correctly"
else
    echo "   ℹ️  Using IP address directly (no DNS needed)"
fi
echo ""

echo "=========================================="
echo "Troubleshooting Steps:"
echo "=========================================="
echo ""
echo "If port ${PORT} is NOT open, try these on your VM:"
echo ""
echo "1. Check if server is running:"
echo "   ssh user@${SERVER_IP}"
echo "   ps aux | grep run_server_only"
echo ""
echo "2. Check if port is listening on correct interface:"
echo "   netstat -tulpn | grep ${PORT}"
echo "   # Should show: 0.0.0.0:${PORT} (NOT 127.0.0.1:${PORT})"
echo ""
echo "3. Open firewall (Ubuntu/Debian):"
echo "   sudo ufw allow ${PORT}/tcp"
echo "   sudo ufw status"
echo ""
echo "4. Open firewall (RedHat/CentOS):"
echo "   sudo firewall-cmd --permanent --add-port=${PORT}/tcp"
echo "   sudo firewall-cmd --reload"
echo ""
echo "5. If using cloud VM (AWS/Azure/GCP):"
echo "   - Check Security Group / NSG / Firewall Rules in cloud console"
echo "   - Add inbound rule: TCP port ${PORT} from 0.0.0.0/0"
echo ""
echo "6. Start server correctly:"
echo "   python run_server_only.py \\"
echo "     --config experiments/configs/baseline_experiment.yaml \\"
echo "     --host 0.0.0.0 --port ${PORT}"
echo ""
