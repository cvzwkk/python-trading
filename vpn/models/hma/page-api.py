# live_table_ngrok_fixed.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import requests
from pyngrok import ngrok, conf
import uvicorn

# -----------------------------
# CONFIGURE VARIABLES HERE
# -----------------------------
NGROK_AUTH_TOKEN = "36xkALQDnxGLwLU3o1CIo2SKsvt_7cUEHiQnMbNC2Snv5bfKk"  # <-- replace with your ngrok token
NGROK_DASHBOARD_PORT = 4041                       # <-- change dashboard port if needed
LOCAL_PORT = 8080 # <-- FastAPI server port

# Set ngrok auth token
if NGROK_AUTH_TOKEN:
    conf.get_default().auth_token = NGROK_AUTH_TOKEN

# Set ngrok dashboard port
conf.get_default().ngrok_port = NGROK_DASHBOARD_PORT

# API endpoint to fetch live data
API_URL = "https://tiesha-nonfissile-jarvis.ngrok-free.dev/live"

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

# HTML page with live-updating table
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Live Trading Table</title>
<style>
  body { font-family: Arial; margin: 20px; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
  th { background-color: #f4f4f4; }
  .negative { color: red; }
  .positive { color: green; }
</style>
</head>
<body>
<h2>Live Trading Data</h2>
<p>Last updated: <span id="timestamp">-</span></p>
<p>Balance: <span id="balance">-</span> | Total PnL: <span id="total_pnl">-</span></p>
<table id="liveTable">
  <thead>
    <tr>
      <th>Exchange</th>
      <th>Price</th>
      <th>Prediction</th>
      <th>Position</th>
      <th>PnL</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>

<script>
async function updateTable() {
  try {
    const res = await fetch('/data');
    const data = await res.json();

    document.getElementById('timestamp').textContent = data.timestamp;
    document.getElementById('balance').textContent = data.balance.toFixed(2);
    document.getElementById('total_pnl').textContent = data.total_pnl.toFixed(2);
    document.getElementById('total_pnl').className = data.total_pnl >= 0 ? 'positive' : 'negative';

    const tbody = document.querySelector('#liveTable tbody');
    tbody.innerHTML = '';

    for (const [exchange, info] of Object.entries(data.exchanges)) {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${exchange}</td>
        <td>${info.price.toFixed(2)}</td>
        <td>${info.prediction.toFixed(2)}</td>
        <td>${info.position}</td>
        <td class="${info.pnl >= 0 ? 'positive' : 'negative'}">${info.pnl.toFixed(6)}</td>
      `;
      tbody.appendChild(row);
    }
  } catch(err) { console.error(err); }
}

// Update every second
setInterval(updateTable, 1000);
updateTable();
</script>
</body>
</html>
"""

# Route for HTML page
@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE

# Route for live API data
@app.get("/data")
def get_data():
    try:
        r = requests.get(API_URL, timeout=5)
        return r.json()
    except:
        return {"timestamp":"-", "balance":0, "total_pnl":0, "exchanges":{}}

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Open ngrok tunnel (HTTP) on LOCAL_PORT, dashboard on NGROK_DASHBOARD_PORT
    public_url = ngrok.connect(addr=LOCAL_PORT, bind_tls=True)
    print(f"Public URL: {public_url}")
    print(f"Ngrok dashboard port: {NGROK_DASHBOARD_PORT}")

    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=LOCAL_PORT)
