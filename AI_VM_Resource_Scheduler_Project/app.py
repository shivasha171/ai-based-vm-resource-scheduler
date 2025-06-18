from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import pandas as pd

app = Flask(__name__)
CORS(app)

# Dummy data for training
data = pd.DataFrame({
    'cpu_usage': [10, 20, 30, 40, 50, 60],
    'memory_usage': [1, 2, 3, 4, 5, 6],
    'future_cpu': [15, 25, 35, 45, 55, 65],
    'future_memory': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
})

# Train models
cpu_model = LinearRegression().fit(data[['cpu_usage']], data['future_cpu'])
mem_model = LinearRegression().fit(data[['memory_usage']], data['future_memory'])

# In-memory storage
vms = []
hosts = [{'host_id': i, 'used_cpu': 0.0, 'used_memory': 0.0} for i in range(1, 4)]
HOST_CPU_LIMIT = 100.0
HOST_MEMORY_LIMIT = 16.0

@app.route('/')
def home():
    return send_file('index.html')  # Serve the frontend

@app.route('/predict', methods=['POST'])
def predict_and_allocate():
    input_data = request.get_json()
    cpu = input_data['cpu_usage']
    memory = input_data['memory_usage']

    predicted_cpu = cpu_model.predict([[cpu]])[0]
    predicted_memory = mem_model.predict([[memory]])[0]

    allocated_host = None
    for host in hosts:
        if (host['used_cpu'] + predicted_cpu <= HOST_CPU_LIMIT and
            host['used_memory'] + predicted_memory <= HOST_MEMORY_LIMIT):
            host['used_cpu'] += predicted_cpu
            host['used_memory'] += predicted_memory
            allocated_host = host['host_id']
            break

    vm_id = len(vms) + 1
    vms.append({
        'vm_id': vm_id,
        'cpu': predicted_cpu,
        'memory': predicted_memory,
        'host_id': allocated_host
    })

    return jsonify({
        'vm_id': vm_id,
        'predicted_cpu': predicted_cpu,
        'predicted_memory': predicted_memory,
        'allocated_host': allocated_host
    })

@app.route('/hosts', methods=['GET'])
def get_hosts():
    return jsonify(hosts)

if __name__ == '__main__':
    # Run on all interfaces (for Docker, VM, or LAN), use port 8000
    app.run(host='0.0.0.0', port=8000, debug=True)
