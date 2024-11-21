import psutil
import socket
import subprocess
import time
import sqlite3
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import os


def resolve_ip(ip):
    try:
        return socket.gethostbyaddr(ip)[0]
    except socket.herror:
        return ip

def estimate_signal_dBm(signal_percentage):
    signal_dBm = signal_percentage * (50 / 100) - 100
    return signal_dBm

def estimate_noise_level(signal_percentage):
    noise_dBm = -100 + (signal_percentage * 50 / 100)
    return noise_dBm
def get_signal_noise():
    try:
        result = subprocess.check_output(['netsh', 'wlan', 'show', 'interfaces']).decode('utf-8')
        signal_strength = None
        noise_level = None

        for line in result.split('\n'):
            if 'RSSI' in line:
                signal_strength = int(line.split(':')[1].strip())
            if 'Noise' in line:
                noise_level = int(line.split(':')[1].strip())
            if 'Signal' in line:
                signal_strength_percent = int(line.split(':')[1].strip().replace('%', ''))
                signal_strength = estimate_signal_dBm(signal_strength_percent)
                noise_level = estimate_noise_level(signal_strength_percent)

        return signal_strength, noise_level
    except Exception as e:
        print(f"Error getting RSSI, Signal, Noise: {e}")
        return None, None

def calculate_snr(signal_strength, noise_level):
    if signal_strength is not None and noise_level is not None:
        P_signal = 10 ** (signal_strength / 10)
        P_noise = 10 ** (noise_level / 10)

        if P_noise == 0:
            return float('inf')
        snr_value = 10 * (P_signal / P_noise)
        return snr_value
    return None

def get_network_stats(duration=10):
    ip_stats = {}

    connections = psutil.net_connections(kind='inet')
    initial_counters = psutil.net_io_counters(pernic=True)

    active_interfaces = {nic: stats for nic, stats in initial_counters.items() if
                         stats.bytes_sent > 0 or stats.bytes_recv > 0}

    for conn in connections:
        if conn.raddr:
            remote_ip = conn.raddr.ip
            if remote_ip not in ip_stats:
                ip_stats[remote_ip] = {'sent': 0, 'recv': 0, 'sent_bytes': 0, 'recv_bytes': 0, 'snr': 0}

    time.sleep(duration)

    final_counters = psutil.net_io_counters(pernic=True)

    for conn in connections:
        if conn.raddr:
            remote_ip = conn.raddr.ip
            if remote_ip in ip_stats:
                for nic in active_interfaces:
                    sent_data = final_counters[nic].bytes_sent - initial_counters[nic].bytes_sent
                    recv_data = final_counters[nic].bytes_recv - initial_counters[nic].bytes_recv
                    ip_stats[remote_ip]['sent_bytes'] += sent_data
                    ip_stats[remote_ip]['recv_bytes'] += recv_data

                ip_stats[remote_ip]['sent'] += 1
                ip_stats[remote_ip]['recv'] += 1

                signal_strength, noise_level = get_signal_noise()
                ip_stats[remote_ip]['snr'] = calculate_snr(signal_strength, noise_level)

    return ip_stats
def save_to_db_and_print(ip_stats):
    directory = r"C:\Users\mulle\Desktop\FTDK_NetworkMonitoring\networkmonitoring\recently"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    db_filename = os.path.join(directory, f"network_stats{timestamp}.db")

    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS ip_stats (
                        ip TEXT,
                        sent_packets INTEGER,
                        sent_data REAL,
                        bandwidth_sent REAL,
                        recv_packets INTEGER,
                        recv_data REAL,
                        bandwidth_recv REAL,
                        packet_loss REAL,
                        snr REAL)''')

    for ip, stats in ip_stats.items():
        sent_kb = stats['sent_bytes'] / 1024
        recv_kb = stats['recv_bytes'] / 1024
        packet_loss = 0

        if stats['sent'] > 0:
            packet_loss = ((stats['sent'] - stats['recv']) / stats['sent']) * 100

        bandwidth_sent = sent_kb / 10
        bandwidth_recv = recv_kb / 10


        cursor.execute('''INSERT INTO ip_stats (ip, sent_packets, sent_data, bandwidth_sent, recv_packets, recv_data, bandwidth_recv, packet_loss, snr)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (ip, stats['sent'], sent_kb, bandwidth_sent, stats['recv'], recv_kb, bandwidth_recv, packet_loss,
                        stats['snr']))

        print(f"IP Cím: {ip}")
        print(f"  Küldött Csomagok: {stats['sent']}, Küldött Adat: {sent_kb:.2f} KB, Sávszélesség: {bandwidth_sent:.2f} KB/s")
        print(
            f"  Érkezett Csomagok: {stats['recv']}, Érkezett Adat: {recv_kb:.2f} KB, Sávszélesség: {bandwidth_recv:.2f} KB/s")
        print(f"  Csomagvesztés: {packet_loss:.2f}%, SNR: {stats['snr']:.2f}")

    conn.commit()
    conn.close()

    print(f"\n[MAIN] Adatok mentése sikeres volt. Fájlnév: {db_filename} /// Data saved to {db_filename}\n")
def start_monitoring(duration):
    print(f"[MAIN] A Hálózat Elemző elindult! /// Network Monitoring started!\n"
          f"[MAIN] Hálózati forgalom figyelése {duration} másodpercig... /// Monitoring network traffic for {duration} seconds...\n")

    ip_stats = get_network_stats(duration=duration)
    save_to_db_and_print(ip_stats)

def create_gui():
    root = tk.Tk()
    root.title("Network Monitor")
    root.geometry("500x250")
    root.configure(bg="#F0F0F0")

    title_label = tk.Label(root, text="Hálózat Elemző", font=("Arial", 24), bg="#F0F0F0", fg="#333333")
    title_label.pack(pady=20)

    tk.Label(root, text="Kérem adja meg az elemzés tartamát (másodpercekben):", font=("Arial", 12), bg="#F0F0F0", fg="#555555").pack(pady=10)

    duration_entry = tk.Entry(root, font=("Arial", 12), width=10)
    duration_entry.pack(pady=10)

    def on_start():
        try:
            duration = int(duration_entry.get())
            start_monitoring(duration)
            messagebox.showinfo("A Folyamat Sikeresen Befejeződött", f"A hálózat elemézése befejeződött {duration} másodperc alatt! Az adatok sikeresen el lettek mentve.")
            run_results_script()
        except ValueError:
            messagebox.showerror("Érvénytelen Karakter", "Kérem adjon meg egy érvényes számot a tartamhoz!")

    start_button = tk.Button(root, text="Folyamat Indítása", command=on_start, font=("Arial", 12), bg="#181818", fg="#FFFFFF", padx=10, pady=5, borderwidth=0, relief="flat")
    start_button.pack(pady=20)

    root.mainloop()
def run_results_script():
    try:
        subprocess.run(["python", "results.py"], check=True)
        print("[MAIN] results.py futása sikeresen befejeződött.")
    except subprocess.CalledProcessError as e:
        print(f"[MAIN] results.py futása hiba miatt megszakadt: {e}")

create_gui()

def calculate_snr(signal_strength_dBm, noise_level_dBm):
    if signal_strength_dBm is not None and noise_level_dBm is not None:
        return signal_strength_dBm - noise_level_dBm
    return None