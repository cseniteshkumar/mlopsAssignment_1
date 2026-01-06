"""
Lightweight Prometheus metrics exporter for server tracking (CPU, memory, disk, network, load, uptime).
- Expose metrics on /metrics for Prometheus to scrape.
- Grafana can use Prometheus as a data source and build dashboards from these metrics.

Requirements:
  pip install prometheus_client psutil

Run:
  python src/grafana_metrics_exporter.py --port 9100 --interval 5
"""
import time
import argparse
import threading
import platform
from prometheus_client import start_http_server, Gauge
try:
    import psutil
except ImportError:
    raise SystemExit("psutil required: pip install psutil")

# Gauges
GAUGE_CPU_PERCENT = Gauge("server_cpu_percent", "CPU usage percent (0-100)")
GAUGE_CPU_COUNT = Gauge("server_cpu_count", "Logical CPU count")
GAUGE_MEM_TOTAL = Gauge("server_memory_total_bytes", "Total memory in bytes")
GAUGE_MEM_USED = Gauge("server_memory_used_bytes", "Used memory in bytes")
GAUGE_MEM_PERCENT = Gauge("server_memory_percent", "Memory usage percent (0-100)")
GAUGE_DISK_TOTAL = Gauge("server_disk_total_bytes", "Total disk bytes for root")
GAUGE_DISK_USED = Gauge("server_disk_used_bytes", "Used disk bytes for root")
GAUGE_DISK_PERCENT = Gauge("server_disk_percent", "Disk usage percent (0-100)")
GAUGE_NET_BYTES_SENT = Gauge("server_network_bytes_sent_total", "Total network bytes sent")
GAUGE_NET_BYTES_RECV = Gauge("server_network_bytes_recv_total", "Total network bytes received")
GAUGE_LOAD_1 = Gauge("server_load_1", "1-minute load average (Unix)")
GAUGE_LOAD_5 = Gauge("server_load_5", "5-minute load average (Unix)")
GAUGE_LOAD_15 = Gauge("server_load_15", "15-minute load average (Unix)")
GAUGE_UPTIME_SECONDS = Gauge("server_uptime_seconds", "System uptime in seconds")
GAUGE_PLATFORM = Gauge("server_platform_info", "Platform info as labeled metric", ['system', 'node', 'release', 'version', 'machine', 'processor'])

_prev_net = None


def collect_once():
    global _prev_net
    # CPU
    GAUGE_CPU_PERCENT.set(psutil.cpu_percent(interval=None))
    GAUGE_CPU_COUNT.set(psutil.cpu_count(logical=True) or 0)

    # Memory
    vm = psutil.virtual_memory()
    GAUGE_MEM_TOTAL.set(vm.total)
    GAUGE_MEM_USED.set(vm.used)
    GAUGE_MEM_PERCENT.set(vm.percent)

    # Disk (root)
    try:
        du = psutil.disk_usage('/')
        GAUGE_DISK_TOTAL.set(du.total)
        GAUGE_DISK_USED.set(du.used)
        GAUGE_DISK_PERCENT.set(du.percent)
    except Exception:
        # Fallback: no root mount available
        pass

    # Network counters (total)
    net_io = psutil.net_io_counters(pernic=False)
    if net_io:
        GAUGE_NET_BYTES_SENT.set(net_io.bytes_sent)
        GAUGE_NET_BYTES_RECV.set(net_io.bytes_recv)

    # Load (Unix; on Windows will raise)
    try:
        load1, load5, load15 = psutil.getloadavg()
        GAUGE_LOAD_1.set(load1)
        GAUGE_LOAD_5.set(load5)
        GAUGE_LOAD_15.set(load15)
    except (AttributeError, OSError):
        # Not available on Windows; set to 0
        GAUGE_LOAD_1.set(0)
        GAUGE_LOAD_5.set(0)
        GAUGE_LOAD_15.set(0)

    # Uptime
    try:
        boot = psutil.boot_time()
        GAUGE_UPTIME_SECONDS.set(max(0.0, time.time() - boot))
    except Exception:
        pass

    # Platform info exposed as metric with labels
    info = {
        'system': platform.system(),
        'node': platform.node(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor() or ""
    }
    # Set a constant value (1) with labels so Grafana/Prometheus can filter by them.
    GAUGE_PLATFORM.labels(**info).set(1)


def loop(interval):
    while True:
        try:
            collect_once()
        except Exception:
            # keep exporter resilient; don't crash on metric collection errors
            pass
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Prometheus exporter for basic server metrics")
    parser.add_argument("--port", type=int, default=9100, help="HTTP port to expose /metrics (default: 9100)")
    parser.add_argument("--interval", type=int, default=5, help="Collection interval in seconds (default: 5)")
    args = parser.parse_args()

    start_http_server(args.port)
    t = threading.Thread(target=loop, args=(args.interval,), daemon=True)
    t.start()
    # Keep main thread alive
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()