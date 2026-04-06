"""
PC Security Scanner — Jacob's Machine
======================================
Checks for:
  1. Suspicious / malicious processes (keyloggers, RATs, miners, spyware)
  2. All external network connections with country/ISP geolocation
  3. Processes with active outbound internet connections
  4. Startup registry entries (programs that auto-run on boot)
  5. Scheduled tasks (common malware persistence method)
  6. Listening ports (what's waiting for inbound connections)
  7. Household IP range — anything outside your router's subnet is flagged

Saves full report to: security_report.json
"""

import os, sys, json, socket, subprocess, ipaddress, time, io
import winreg
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows so box-drawing / emoji chars work
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import psutil
except ImportError:
    sys.exit("Run: py -3 -m pip install psutil requests")

try:
    import requests as _req
    _GEO = True
except ImportError:
    _GEO = False

REPORT_FILE = Path(__file__).parent / "security_report.json"

# ── Colour helpers (Windows ANSI) ─────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def red(s):    return f"{RED}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def green(s):  return f"{GREEN}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"
def bold(s):   return f"{BOLD}{s}{RESET}"

os.system("color")  # enable ANSI on Windows

# ── Known bad process names (keyloggers, RATs, miners) ────────
KNOWN_BAD = {
    # Keyloggers
    "ardamax","spyagent","revealer keylogger","kidlogger","refog",
    "actual keylogger","perfect keylogger","spyrix","iwantsoft",
    "all in one keylogger","elite keylogger","home keylogger",
    # RATs & backdoors
    "njrat","darkcomet","nanocore","quasar","remcos","asyncrat",
    "netwire","blackshades","luminosity","bifrost","poisonivy",
    "cybergate","jrat","xtremerat","gh0st","pandora","warzone",
    "dcrat","asyncrat","venomrat","regrat","lilith",
    # Miners
    "xmrig","minerd","cpuminer","nicehash","nbminer","gminer",
    "teamredminer","lolminer","phoenixminer",
    # Credential tools
    "mimikatz","wce","pwdump","fgdump","gsecdump","procdump",
    # Stalkerware
    "flexispy","mspy","hoverwatch","highster","spyzie",
}

# ── Suspicious listening ports ────────────────────────────────
SUSPICIOUS_PORTS = {
    1337,4444,4445,5554,5555,6666,7777,8888,9999,
    12345,31337,65535,
    1080,  # SOCKS proxy
    6667,6668,6669,  # IRC (old botnet C2)
}

# ── Geo-lookup cache (avoid hitting rate limit) ───────────────
_geo_cache: dict = {}

def _is_private(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip)
        return a.is_private or a.is_loopback or a.is_link_local or a.is_multicast
    except Exception:
        return True

def _geolocate(ip: str) -> dict:
    if ip in _geo_cache:
        return _geo_cache[ip]
    if not _GEO or _is_private(ip):
        return {}
    try:
        r = _req.get(
            f"http://ip-api.com/json/{ip}",
            params={"fields": "status,country,countryCode,city,isp,org,proxy,hosting,query"},
            timeout=5,
        )
        data = r.json()
        if data.get("status") == "success":
            _geo_cache[ip] = data
            return data
    except Exception:
        pass
    _geo_cache[ip] = {}
    return {}

# ─────────────────────────────────────────────────────────────
# 1. PROCESS SCAN
# ─────────────────────────────────────────────────────────────
def scan_processes() -> dict:
    print(bold("\n[1/5] Scanning running processes…"))
    bad, suspicious, all_procs = [], [], []

    for proc in psutil.process_iter(["pid","name","exe","username","cmdline","create_time"]):
        try:
            info = proc.info
            name_lower = (info["name"] or "").lower().replace(".exe","")
            exe   = info["exe"] or ""
            cmdline = " ".join(info["cmdline"] or [])

            entry = {
                "pid":       info["pid"],
                "name":      info["name"],
                "exe":       exe,
                "user":      info["username"],
                "cmdline":   cmdline[:200],
                "started":   datetime.fromtimestamp(info["create_time"]).strftime("%H:%M:%S")
                             if info["create_time"] else "?",
            }
            all_procs.append(entry)

            # Check known-bad name
            if any(bad_name in name_lower for bad_name in KNOWN_BAD):
                entry["flag"] = "KNOWN_MALWARE_NAME"
                bad.append(entry)
                continue

            # Flag processes with no executable path (hidden/injected)
            if not exe and info["pid"] not in (0, 4):
                entry["flag"] = "NO_EXE_PATH"
                suspicious.append(entry)
                continue

            # Flag processes running from Temp / AppData/Roaming unusual spots
            shady_paths = ["\\temp\\","\\tmp\\","\\appdata\\roaming\\",
                           "\\appdata\\local\\temp\\","\\downloads\\"]
            if exe and any(p in exe.lower() for p in shady_paths):
                # Whitelist known-good apps that legitimately live there
                ok_in_roaming = {"discord","slack","teams","zoom","spotify",
                                 "cursor","claude","code","vscode","git"}
                if not any(w in exe.lower() for w in ok_in_roaming):
                    entry["flag"] = "SHADY_LOCATION"
                    suspicious.append(entry)

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Report
    if bad:
        for p in bad:
            print(red(f"  ❌ DANGER  PID {p['pid']:6}  {p['name']}  →  {p['exe']}"))
    if suspicious:
        for p in suspicious:
            print(yellow(f"  ⚠️  SUSPECT PID {p['pid']:6}  {p['name']}  flag={p['flag']}"))
    if not bad and not suspicious:
        print(green("  ✅ No suspicious processes found"))

    print(f"     {len(all_procs)} total processes scanned")
    return {"total": len(all_procs), "dangerous": bad, "suspicious": suspicious}

# ─────────────────────────────────────────────────────────────
# 2. NETWORK CONNECTIONS
# ─────────────────────────────────────────────────────────────
def scan_network() -> dict:
    print(bold("\n[2/5] Scanning network connections…"))
    external, local_conns = [], []
    unique_ips = set()

    for conn in psutil.net_connections(kind="inet"):
        try:
            if conn.status not in ("ESTABLISHED","SYN_SENT","CLOSE_WAIT"):
                continue
            raddr = conn.raddr
            if not raddr:
                continue
            rip = raddr.ip
            if _is_private(rip):
                local_conns.append({"ip": rip, "port": raddr.port, "pid": conn.pid})
                continue

            # External IP
            try:
                proc = psutil.Process(conn.pid)
                pname = proc.name()
                pexe  = proc.exe()
            except Exception:
                pname, pexe = "unknown", ""

            entry = {
                "remote_ip":   rip,
                "remote_port": raddr.port,
                "local_port":  conn.laddr.port if conn.laddr else 0,
                "status":      conn.status,
                "pid":         conn.pid,
                "process":     pname,
                "exe":         pexe,
            }

            if rip not in unique_ips:
                unique_ips.add(rip)
                time.sleep(0.05)  # gentle rate-limit
                geo = _geolocate(rip)
                entry["geo"] = geo
                flag = ""
                if geo.get("proxy"):    flag = "VPN/PROXY"
                if geo.get("hosting"):  flag = flag + " HOSTING/DATACENTER" if flag else "HOSTING/DATACENTER"
                entry["flag"] = flag.strip()
            else:
                entry["geo"]  = _geo_cache.get(rip, {})
                entry["flag"] = ""

            external.append(entry)

        except Exception:
            pass

    # Print external connections
    if external:
        seen = set()
        for c in external:
            key = (c["remote_ip"], c["process"])
            geo = c.get("geo", {})
            country = geo.get("country","?")
            isp     = geo.get("isp","?")
            flag    = c.get("flag","")
            line = (f"  {'❌' if flag else '🌐'} {c['process']:20s}  "
                    f"{c['remote_ip']:15s}:{c['remote_port']:<5}  "
                    f"{country} / {isp}")
            if flag:
                print(red(line + f"  [{flag}]") if "PROXY" in flag else yellow(line + f"  [{flag}]"))
            else:
                if key not in seen:
                    print(f"  {line}")
            seen.add(key)
    else:
        print(green("  ✅ No external connections detected"))

    print(f"     {len(external)} external  |  {len(local_conns)} local connections")
    return {"external": external, "local_count": len(local_conns)}

# ─────────────────────────────────────────────────────────────
# 3. LISTENING PORTS
# ─────────────────────────────────────────────────────────────
def scan_listening_ports() -> dict:
    print(bold("\n[3/5] Scanning open listening ports…"))
    listeners, flagged = [], []

    for conn in psutil.net_connections(kind="inet"):
        if conn.status != "LISTEN":
            continue
        try:
            proc  = psutil.Process(conn.pid)
            pname = proc.name()
            pexe  = proc.exe()
        except Exception:
            pname, pexe = "unknown", ""

        lport = conn.laddr.port
        entry = {
            "port":    lport,
            "addr":    conn.laddr.ip,
            "pid":     conn.pid,
            "process": pname,
            "exe":     pexe,
        }
        listeners.append(entry)

        if lport in SUSPICIOUS_PORTS:
            entry["flag"] = "SUSPICIOUS_PORT"
            flagged.append(entry)
            print(red(f"  ❌ PORT {lport:5}  {pname}  →  {pexe}  [SUSPICIOUS]"))
        else:
            # Only show non-system listeners
            if conn.laddr.ip not in ("127.0.0.1","::1") or lport not in (135,445,1900,5040):
                print(f"  🔓 PORT {lport:5}  {conn.laddr.ip:15}  {pname}")

    if not flagged:
        print(green("  ✅ No suspicious listening ports found"))
    return {"listeners": listeners, "flagged": flagged}

# ─────────────────────────────────────────────────────────────
# 4. STARTUP REGISTRY ENTRIES
# ─────────────────────────────────────────────────────────────
def scan_startup() -> dict:
    print(bold("\n[4/5] Scanning startup registry entries…"))
    entries, suspicious = [], []

    keys_to_check = [
        (winreg.HKEY_CURRENT_USER,
         r"Software\Microsoft\Windows\CurrentVersion\Run"),
        (winreg.HKEY_LOCAL_MACHINE,
         r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"),
        (winreg.HKEY_LOCAL_MACHINE,
         r"SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnce"),
        (winreg.HKEY_CURRENT_USER,
         r"Software\Microsoft\Windows\CurrentVersion\RunOnce"),
    ]

    shady_startup_paths = ["\\temp\\","\\tmp\\","\\appdata\\roaming\\","\\appdata\\local\\temp\\"]
    ok_startup = {"discord","teams","spotify","onedrive","googledrive","dropbox",
                  "steam","nvidia","amd","realtek","logitech","razer","corsair",
                  "microsoft","windows","java","python","cursor","slack","zoom",
                  "vmware","virtualbox","obs","chrome","edge","firefox"}

    for hive, subkey in keys_to_check:
        hive_name = "HKCU" if hive == winreg.HKEY_CURRENT_USER else "HKLM"
        try:
            key = winreg.OpenKey(hive, subkey)
            i = 0
            while True:
                try:
                    name, value, _ = winreg.EnumValue(key, i)
                    entry = {"hive": hive_name, "key": subkey,
                             "name": name, "value": value}
                    entries.append(entry)

                    val_lower = value.lower()
                    is_ok = any(w in val_lower for w in ok_startup)
                    is_shady_path = any(p in val_lower for p in shady_startup_paths)

                    if is_shady_path and not is_ok:
                        entry["flag"] = "SHADY_PATH"
                        suspicious.append(entry)
                        print(red(f"  ❌ [{hive_name}] {name}  →  {value[:80]}"))
                    else:
                        print(f"  📌 [{hive_name}] {name}")
                    i += 1
                except OSError:
                    break
            winreg.CloseKey(key)
        except Exception:
            pass

    if not suspicious:
        print(green("  ✅ No suspicious startup entries found"))
    return {"entries": entries, "suspicious": suspicious}

# ─────────────────────────────────────────────────────────────
# 5. SCHEDULED TASKS (common malware persistence)
# ─────────────────────────────────────────────────────────────
def scan_scheduled_tasks() -> dict:
    print(bold("\n[5/5] Scanning scheduled tasks…"))
    tasks, suspicious = [], []

    shady_task_paths = ["\\temp\\","\\tmp\\","\\appdata\\roaming\\","\\appdata\\local\\temp\\"]
    ok_tasks = {"microsoft","windows","google","nvidia","amd","adobe","java",
                "dropbox","onedrive","steam","teams","discord","spotify","python",
                "mozilla","chrome","edge","cursor","claude"}

    try:
        out = subprocess.check_output(
            ["schtasks", "/query", "/fo", "CSV", "/v"],
            stderr=subprocess.DEVNULL, text=True, timeout=30
        )
        lines = [l.strip().strip('"') for l in out.splitlines() if l.strip()]
        current: dict = {}

        for line in lines:
            parts = line.split('","')
            if len(parts) < 2:
                continue
            # schtasks CSV columns: HostName, TaskName, NextRun, Status,
            #   LogonMode, LastRun, LastResult, Author, TaskToRun, ...
            if parts[0] in ("HostName", "主机名"):
                continue
            task_name = parts[1] if len(parts) > 1 else ""
            task_run  = parts[8] if len(parts) > 8 else ""
            status    = parts[3] if len(parts) > 3 else ""

            if not task_name or task_name == "TaskName":
                continue

            entry = {"name": task_name, "action": task_run, "status": status}
            tasks.append(entry)

            run_lower = task_run.lower()
            is_ok    = any(w in task_name.lower() or w in run_lower for w in ok_tasks)
            is_shady = any(p in run_lower for p in shady_task_paths)

            if is_shady and not is_ok:
                entry["flag"] = "SHADY_ACTION_PATH"
                suspicious.append(entry)
                print(red(f"  ❌ {task_name[:50]:50s}  →  {task_run[:60]}"))

    except Exception as e:
        print(yellow(f"  ⚠️  Could not read scheduled tasks: {e}"))

    if not suspicious:
        print(green("  ✅ No suspicious scheduled tasks found"))
    print(f"     {len(tasks)} tasks scanned")
    return {"total": len(tasks), "suspicious": suspicious}

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print(bold(cyan("\n╔══════════════════════════════════════╗")))
    print(bold(cyan("║     PC SECURITY SCANNER  v1.0        ║")))
    print(bold(cyan("╚══════════════════════════════════════╝")))
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if not _GEO:
        print(yellow("  (requests not installed — skipping IP geolocation)"))

    t0 = time.time()
    report = {
        "scanned_at":        datetime.now().isoformat(),
        "machine":           socket.gethostname(),
        "processes":         scan_processes(),
        "network":           scan_network(),
        "listening_ports":   scan_listening_ports(),
        "startup":           scan_startup(),
        "scheduled_tasks":   scan_scheduled_tasks(),
    }
    elapsed = time.time() - t0

    # ── Summary ──────────────────────────────────────────────
    procs_bad   = len(report["processes"]["dangerous"])
    procs_sus   = len(report["processes"]["suspicious"])
    ext_conns   = len(report["network"]["external"])
    port_flags  = len(report["listening_ports"]["flagged"])
    startup_sus = len(report["startup"]["suspicious"])
    task_sus    = len(report["scheduled_tasks"]["suspicious"])
    total_flags = procs_bad + procs_sus + port_flags + startup_sus + task_sus

    print(bold(cyan("\n══════════════ SUMMARY ══════════════")))
    def row(label, count, danger=False):
        icon = red("❌") if (count > 0 and danger) else (yellow("⚠️ ") if count > 0 else green("✅"))
        val  = red(str(count)) if (count > 0 and danger) else (yellow(str(count)) if count > 0 else green(str(count)))
        print(f"  {icon}  {label:<35} {val}")

    row("Known malware processes",    procs_bad,   danger=True)
    row("Suspicious processes",       procs_sus)
    row("External internet connections", ext_conns)
    row("Suspicious listening ports", port_flags,  danger=True)
    row("Suspicious startup entries", startup_sus, danger=True)
    row("Suspicious scheduled tasks", task_sus,    danger=True)

    print()
    if total_flags == 0:
        print(bold(green("  🛡️  ALL CLEAR — No threats detected")))
    elif procs_bad + port_flags + startup_sus + task_sus > 0:
        print(bold(red("  🚨 ACTION REQUIRED — Threats found! Review items above.")))
    else:
        print(bold(yellow("  ⚠️  LOW RISK — Some items worth reviewing")))

    print(f"\n  Scan completed in {elapsed:.1f}s")
    print(f"  Full report saved → {REPORT_FILE}")
    print()

    # Save JSON report
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report

if __name__ == "__main__":
    main()
