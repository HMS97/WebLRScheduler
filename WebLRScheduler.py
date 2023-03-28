import torch.optim.lr_scheduler as lr_scheduler
from werkzeug.serving import make_server
from functools import partial
from flask import Flask, request, jsonify, render_template_string
import mplcursors
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import seaborn as sns
import threading
import math
import logging
import os 
sns.set_style('darkgrid')
os.environ['FLASK_ENV'] = 'production'

class WebLRScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_lambda = None, port =5000, gamma = 0.999, total_iteration=None,last_epoch = -1):

        self.num_training_steps = total_iteration
        self.app = Flask(__name__)
        self.app.config['PROPAGATE_EXCEPTIONS'] = True
        self.port = port
        log = logging.getLogger('werkzeug')
        log_handler = logging.FileHandler('LRSchedulerServer.log')
        log.addHandler(log_handler)

        self.app.logger.setLevel(logging.ERROR)
        self.scheduler_type = None
        self.optimizer = optimizer
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.initial_learning_rate = optimizer.param_groups[0]['lr']
        if lr_lambda is None:
            self.scheduler_type = 'cosine'
            self.reset_cosine_scheduler(self.initial_learning_rate)
        else:
            self.lr_lambda = lr_lambda
        self.lr_history = []
        self.count = 0

        super().__init__(optimizer,  last_epoch=last_epoch, verbose= False )

        self.app.route('/')(self.index)
        self.app.route('/update_learning_rate', methods=['POST'])(self.update_learning_rate)
        self.app.route('/fetch_learning_rate_history', methods=['GET'])(self.fetch_learning_rate_history)
        self.app.route('/fetch_sample_schedule', methods=['GET'])(self.fetch_sample_schedule)
        self.app.route('/fetch_info', methods=['GET'])(self.fetch_info)
        self.app.route('/fetch_learning_rate_plot', methods=['GET'])(self.fetch_learning_rate_plot)

        self.get_html()
        self.server_thread = threading.Thread(target=self.start)
        self.server_thread.daemon = True
        self.server_thread.start()


    def get_lr(self):
        lr = [self.lr_lambda(self.last_epoch)]
        self.lr_history.append(self.optimizer.param_groups[0]['lr'])
        self.count += 1
        self.get_info()
        return   lr

    def get_info(self):
        self.show_dict = {'gamma': self.gamma, 
                        'total_steps':self.num_training_steps, 
                        'initial_lr': self.initial_learning_rate,
                        'current_lr': round(self.optimizer.param_groups[0]['lr'],7),
                        'scheduler_type': self.scheduler_type,
                        'Steps': self.count,
                        }

    def fetch_info(self):
        return jsonify(self.show_dict)


    def update_learning_rate(self):
        new_learning_rate = request.json.get('learning_rate')
        self.scheduler_type = request.json.get('scheduler')
 

        if new_learning_rate is not None and self.scheduler_type is not None:
            sample_img_data = self.create_sample_learning_rate_plot(1000)
            self.reset_scheduler_with_type(new_learning_rate)

            return jsonify({'message': 'Learning rate and scheduler type updated', 'learning_rate': new_learning_rate, 'scheduler_type': self.scheduler_type, 'sample_img_data': sample_img_data}), 200
        else:
            return jsonify({'message': 'No learning rate provided'}), 400


    def reset_scheduler_with_type(self, new_learning_rate):
        if self.scheduler_type == 'linear':
            self.reset_linear_scheduler(new_learning_rate)
        elif self.scheduler_type == 'cosine':
            self.reset_cosine_scheduler(new_learning_rate)
        elif self.scheduler_type == 'exponential':
            self.reset_exponential_scheduler(new_learning_rate)
        else:
            raise ValueError(f"Invalid scheduler type: {self.scheduler_type}")

    def reset_linear_scheduler(self, new_learning_rate):
        lr_lambda = partial(
            self.get_linear_schedule_with_warmup_lr_lambda,
            lr=new_learning_rate,
            num_warmup_steps=0,
        )
        self.lr_lambda = lr_lambda

    def reset_cosine_scheduler(self, new_learning_rate ):
        lr_lambda = lambda iteeration: new_learning_rate * 0.5 * (1 + math.cos(math.pi * iteeration / self.num_training_steps))
        self.lr_lambda = lr_lambda

    def reset_exponential_scheduler(self, new_learning_rate):
        lr_lambda = lambda iteeration: new_learning_rate * (self.gamma ** iteeration)
        self.lr_lambda = lr_lambda


    def get_linear_schedule_with_warmup_lr_lambda(self, current_step, *, lr, num_warmup_steps ):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(self.num_training_steps - current_step) / float(max(1, self.num_training_steps - num_warmup_steps))) * lr

    def start(self):
        server = make_server('127.0.0.1', self.port, self.app)
        server.serve_forever()

    def fetch_learning_rate_history(self):
        return self.lr_history
    
    def index(self):
        learning_rate_history = self.fetch_learning_rate_history()
        sample_lr = self.get_lr()

        new_lr_img_data = self.create_learning_rate_plot(learning_rate_history)
        sample_img_data = self.create_sample_learning_rate_plot(1000)
        variables = {
                'sample_img_data': sample_img_data,
                'new_lr_img_data': new_lr_img_data,
                'sample_lr': sample_lr[0],
                **self.show_dict, 
            }
        return render_template_string(self.html_template, **variables)

    def fetch_learning_rate_plot(self):
        learning_rate_history = self.fetch_learning_rate_history()
        img_data = self.create_learning_rate_plot(learning_rate_history)
        return jsonify({'img_data': img_data}), 200

  
    def create_learning_rate_plot(self, history):
        plt.figure()
        plt.plot(history)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate History')
        plt.grid()
        mplcursors.cursor(hover=True)
        plt.ylim(bottom= min(0,min(history)), top = max(history) +min(history)/5 )
        plt.grid()

        # Set x-axis limit to start from 0 and step 1
        plt.xlim(left=0)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close()
        return img_data

    def fetch_sample_schedule(self):
        iterations = request.args.get('iterations', type=int)
        if iterations is not None and iterations > 0:
            sample_img_data = self.create_sample_learning_rate_plot(iterations)
            return jsonify({'sample_img_data': sample_img_data}), 200
        else:
            return jsonify({'message': 'Invalid number of iterations'}), 400

    def create_sample_learning_rate_plot(self, iterations):
        sample_history = [self.lr_lambda(step) for step in range(iterations)]
        plt.figure()
        plt.plot(sample_history)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Sample Learning Rate Schedule')
        plt.grid()
        mplcursors.cursor(hover=True)
        plt.ylim(bottom= min(0,min(sample_history)), top = max(sample_history) +min(sample_history)/5 )
        plt.grid()

        plt.xlim(left=0)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.savefig('sample.png', format='png')

        buf.seek(0)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close()
        return img_data
  
    def get_html(self):
        self.html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Learning Rate History</title>
    <style>
            body {
            font-family: Arial, Helvetica, sans-serif;
        }
        .container {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .input-container {
            display: flex;
            align-items: center;
            margin-right: 20px;
        }
        label {
            margin-right: 10px;
        }
        input[type="text"], select {
            padding: 5px;
            margin-right: 10px;
            border-radius: 5px;
            border: none;
            background-color: #f2f2f2;
            font-size: 16px;
        }
        input[type="text"]:focus, select:focus {
            outline: none;
        }
        button[type="button"] {
            padding: 5px 10px;
            border-radius: 5px;
            border: none;
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }
        button[type="button"]:hover {
            background-color: #3e8e41;
        }
        .sample-schedule-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 20px;
        }
        h1, h2 {
            text-align: center;
        }
        img {
            max-width: 100%;
        }
    .info-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: flex-start;
            margin-top: 20px;
        }

        .info-container div {
            display: flex;
            flex-direction: row;
            align-items: center;
            margin-bottom: 10px;
        }

        .info-container span {
            font-size: 16px;
            margin-right: 10px;
            text-align: left;
        }
    </style>
    <script>
        async function updateLearningRate(event) {
            event.preventDefault();
            
            const learningRate = parseFloat(document.getElementById("learningRateInput").value);
            const scheduler = document.getElementById("schedulerSelect").value;
            
            if (!isNaN(learningRate)) {
                const response = await fetch("/update_learning_rate", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ learning_rate: learningRate, scheduler: scheduler })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    document.getElementById("sampleScheduleImage").src = `data:image/png;base64,${data.sample_img_data}`;
                } else {
                    alert("Failed to update learning rate and scheduler type");
                }
            } else {
                alert("Please enter a valid learning rate and select a scheduler type");
            }
        }

    async function fetchSampleSchedule() {
        const iterationsInput = document.getElementById("iterationsInput");
        const iterations = parseInt(iterationsInput.value);

        if (!isNaN(iterations) && iterations > 0) {
            const response = await fetch(`/fetch_sample_schedule?iterations=${iterations}`);

            if (response.ok) {
                const data = await response.json();
                const sampleScheduleImage = document.getElementById("sampleScheduleImage");
                sampleScheduleImage.src = `data:image/png;base64,${data.sample_img_data}`;
            } else {
                alert("Failed to fetch sample schedule");
            }
        } else {
            alert("Please enter a valid number of iterations");
        }
        }

        async function fetchInfo() {
            const response = await fetch(`/fetch_info`);
            if (response.ok) {
                const data = await response.json();
                document.getElementById("gamma").textContent = data.gamma;
                document.getElementById("totalSteps").textContent = data.total_steps;
                document.getElementById("initialLR").textContent = data.initial_lr;
                document.getElementById("CurrentLR").textContent = data.current_lr;
                document.getElementById("schedulerType").textContent = data.scheduler_type;
                document.getElementById("Steps").textContent = data.Steps;

            } else {
                alert("Failed to fetch info");
            }
        }
        
        async function refresh() {
            await fetchInfo();
        }
         async function updateLearningRateHistory() {
        const response = await fetch("/fetch_learning_rate_plot");
        const data = await response.json();
        const imgData = data.img_data;
        document.getElementById("learningRateHistoryImg").src = "data:image/png;base64," + imgData;
    }
         setInterval(fetchInfo, 2000);

         setInterval(updateLearningRateHistory, 2000);
    </script>
</head>
<body onload="refresh()">
    <h1>Learning Rate History</h1>
    <div class="container">
        <div>
            <img id = "learningRateHistoryImg" src="data:image/png;base64,{{ new_lr_img_data }}" alt="Learning Rate History">
        </div>
        <div>
            <div class="sample-schedule-container">
                <img id="sampleScheduleImage" src="data:image/png;base64,{{ sample_img_data }}" alt="Sample Learning Rate Schedule">
                <label for="iterationsInput">Total Iterations: </label>
                <input type="text" id="iterationsInput" placeholder="Enter number of iterations">
                <button type="button" onclick="fetchSampleSchedule()">Generate</button>
            </div>
        </div>
        <div>
         <h2>Info and Set New Learning Rate</h2>
            <div class="info-container">
                <div><span>Gamma:</span><span id="gamma"></span></div>
                <div><span>Initial Learning Rate:</span><span id="initialLR"></span></div>
                <div><span>Current Learning Rate:</span><span id="CurrentLR"></span></div>

                <div><span>Scheduler Type:</span><span id="schedulerType"></span></div>
                <div><span>Total Steps:</span><span id="totalSteps"></span></div>

                <div><span>Steps :</span><span id="Steps"></span></div>

            </div>

            <div class="input-container">
                <label for="learningRateInput">New Learning Rate: </label>
                <input type="text" id="learningRateInput" placeholder="Enter new learning rate">
                <label for="schedulerSelect">Scheduler: </label>
                <select id="schedulerSelect">
                    <option value="linear">Linear</option>
                    <option value="cosine">Cosine</option>
                    <option value="exponential">Exponential</option>
                    <!-- Add more options here for other schedulers -->
                </select>
                <button type="button" onclick="updateLearningRate(event)">Confirm</button>
            </div>
        </div>
    </div>
    
</body>
</html>
"""
