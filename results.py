import http.server
import socketserver
import os
import sqlite3
import webbrowser
import threading
import statistics
import urllib.request
import json
from neuralcalculations import run_neural_calculations
import pandas as pd

print("[RESULTS] A Hálózat Elemező Kiértékelése /// Network Monitoring Results\n")

os.chdir('resultspage')

PORT = 8000

httpd = None

def get_ip_geolocation(ip):
    """Fetch geolocation data for the given IP address using ipapi."""
    try:
        with urllib.request.urlopen(f"https://ipapi.co/{ip}/json/") as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                return data.get("city", "N/A"), data.get("country_name", "N/A")
            else:
                return "Unknown", "Unknown"
    except Exception as e:
        print(f"Error fetching geolocation for {ip}: {e}")
        return "Unknown", "Unknown"

def get_latest_db_file(directory):
    """Get the most recent .db file from the specified directory."""
    db_files = [f for f in os.listdir(directory) if f.endswith('.db')]
    if not db_files:
        return None
    latest_file = max(db_files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return os.path.join(directory, latest_file)

def fetch_data_from_db(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    rows = []
    sent_variance = recv_variance = sent_range = recv_range = average_bandwidth = 0
    packet_differences = []

    try:
        cursor.execute('''SELECT ip, sent_packets, sent_data, bandwidth_sent, 
                                  recv_packets, recv_data, bandwidth_recv, 
                                  packet_loss, snr 
                         FROM ip_stats''')
        rows = cursor.fetchall()

        if rows:
            sent_packets = [row[1] for row in rows]
            recv_packets = [row[4] for row in rows]
            bandwidth_sent = [row[3] for row in rows]
            bandwidth_recv = [row[6] for row in rows]

            # avg bandwidth
            total_bandwidth = sum(bandwidth_sent) + sum(bandwidth_recv)
            average_bandwidth = total_bandwidth / len(rows)

            # spread sent received packets
            sent_variance = statistics.variance(sent_packets) if len(sent_packets) > 1 else 0
            recv_variance = statistics.variance(recv_packets) if len(recv_packets) > 1 else 0

            sent_range = max(sent_packets) - min(sent_packets) if sent_packets else 0
            recv_range = max(recv_packets) - min(recv_packets) if recv_packets else 0

            # difference (sent - received)
            packet_differences = [(row[0], row[1] - row[4]) for row in rows]

    except sqlite3.OperationalError as e:
        print(f"Error: {e}")

    conn.close()
    return rows, sent_variance, recv_variance, sent_range, recv_range, packet_differences, average_bandwidth
class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/user_results.html":
            with open("user_results.html", "r", encoding="utf-8") as file:
                html_content = file.read()

            db_file = get_latest_db_file("C:/Users/mulle/Desktop/FTDK_NetworkMonitoring/networkmonitoring/recently")
            if db_file is None:
                print("[ERROR] No database files found.")
                db_data = []
                average_bandwidth = 0
            else:
                db_data, sent_var, recv_var, sent_rng, recv_rng, pkt_diff, average_bandwidth = fetch_data_from_db(
                    db_file)

            # Run neural
            neural_result_df = run_neural_calculations(
                'C:/Users/mulle/Desktop/FTDK_NetworkMonitoring/networkmonitoring/OUTPUTS_W1',
                db_file
            )

            optimal_ip = neural_result_df.loc[neural_result_df['predicted_packet_loss'].idxmin()]['ip']
            neural_info_html = f"<p>The most optimal IP address is: {optimal_ip}</p>"

            cards_html = ""
            for row in db_data:
                ip_address = row[0]
                sent_packets = row[1]
                recv_packets = row[4]
                bandwidth_sent = row[3]
                bandwidth_recv = row[6]
                city, country = get_ip_geolocation(ip_address)

                # Ellenőrizze, hogy a forgalom nagy, a sávszélesség nagy vagy a teljes sávszélességet lopják
                is_large_traffic = sent_packets > average_bandwidth * 2 or recv_packets > average_bandwidth * 2
                is_stealing_bandwidth = bandwidth_sent > average_bandwidth * 1.5 or bandwidth_recv > average_bandwidth * 1.5
                is_large_bandwidth = bandwidth_sent > average_bandwidth * 1.2 or bandwidth_recv > average_bandwidth * 1.2

                card_color = "#5fde9d"
                if is_large_traffic:
                    card_color = "#f5795b"
                elif is_large_bandwidth or is_stealing_bandwidth:
                    card_color = "#fca44c"

                cards_html += f'''
                        <div class="card" style="background-color:{card_color};">
                            <h3>IP: {ip_address}</h3>
                            <p>Elhelyezkedés: {city}, {country}</p>
                            <p>Küldött Csomagok: {sent_packets}</p>
                            <p>Küldött Adat: {row[2]} bytes</p>
                            <p>Küldött Sávszélesség: {bandwidth_sent} Kbps</p>
                            <p>Érkezett Csomagok: {recv_packets}</p>
                            <p>Érkezett Adat: {row[5]} bytes</p>
                            <p>Érkezett Sávszélesség: {bandwidth_recv} Kbps</p>
                            <p>Csomag Veszteség: {row[7]}%</p>
                            <p>SNR: {row[8]} (dB)</p>
                        </div>
                    '''

            if not db_data:
                cards_html = "<p>Sajnos nincs megjeleníthető adat.</p>"

            calc_html = f'''
                <div class="calculations">
                    <h3>Csomagszórás eredményei:</h3>
                    <p>Küldött Csomagok Varianciája: {sent_var}</p>
                    <p>Érkezett Csomagok Varianciája: {recv_var}</p>
                    <p>Küldött Csomagok Tartománya: {sent_rng}</p>
                    <p>Érkezett Csomagok Tartománya: {recv_rng}</p>
                    <h3>Csomag Különbség (Küldött - Érkezett):</h3>
                    <ul>
                        {''.join([f'<li>{ip}: {diff}</li>' for ip, diff in pkt_diff])}
                    </ul>
                    {neural_info_html}
                </div>
            '''

            neural_results_html = "<h3>Neurális Hálózat Eredményei:</h3><ul>"
            for index, row in neural_result_df.iterrows():
                neural_results_html += f"<li>IP: {row['ip']} - Predicted Packet Loss: {row['predicted_packet_loss']}</li>"
            neural_results_html += "</ul>"

            html_content = html_content.replace(
                '<section id="neuralnetwork"></section>',
                f'''
                <section id="neuralnetwork">
                    <h2>Neurális Hálózat</h2>
                    <p>Itt található a neurális hálózat elemzése és eredményei.</p>
                    {neural_results_html}
                </section>
                '''
            )

            # Replace the cards section directly in the HTML content
            html_content = html_content.replace('<section class="cards"></section>',
                                                f'<section class="cards">{cards_html}</section>')
            html_content = html_content.replace('<div class="calculations"></div>', calc_html)

            print("[RESULTS] A felhasználói eredmények megjelenítve /// User results displayed successfully")

            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html_content.encode('utf-8'))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/shutdown':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Shutting down server...")

            def shutdown_server():
                global httpd
                print("[RESULTS] A szerver leáll /// Shutting down server...")
                httpd.shutdown()

            threading.Thread(target=shutdown_server).start()

with socketserver.TCPServer(("", PORT), MyHandler) as server:

    httpd = server

    url = f"http://localhost:{PORT}/user_results.html"
    webbrowser.open(url)

    print(f"Serving at port {PORT }")
    server.serve_forever()

