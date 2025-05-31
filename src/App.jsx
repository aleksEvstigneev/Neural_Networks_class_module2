import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Paper, Typography, Box, Button, TextField, Slider } from '@mui/material';
import { Line } from 'react-chartjs-2';
import NeuralNetworkViz from './components/NeuralNetworkViz';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import Papa from 'papaparse';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [model, setModel] = useState(null);
  const [inputs, setInputs] = useState({
    timeAlone: 5,
    stageFear: 'No',
    socialEvents: 5,
    goingOutside: 5,
    drainedAfterSocializing: 'No',
    friendsCircle: 7,
    postFrequency: 5
  });
  const [prediction, setPrediction] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [layerActivations, setLayerActivations] = useState([]);
  const [metrics, setMetrics] = useState({
    accuracy: 0,
    precision: 0,
    recall: 0,
    f1Score: 0,
    validationLoss: 0
  });

  useEffect(() => {
    const loadAndPrepareData = async () => {
      const response = await fetch('/personality_dataset.csv');
      const csvData = await response.text();
      
      const results = Papa.parse(csvData, {
        header: true,
        skipEmptyLines: true
      });

      const cleanData = results.data.filter(row => 
        Object.values(row).every(val => val !== '' && val !== undefined)
      );

      const features = cleanData.map(row => [
        parseFloat(row.Time_spent_Alone) / 11.0,
        row.Stage_fear === 'Yes' ? 1 : 0,
        parseFloat(row.Social_event_attendance) / 10.0,
        parseFloat(row.Going_outside) / 7.0,
        row.Drained_after_socializing === 'Yes' ? 1 : 0,
        parseFloat(row.Friends_circle_size) / 15.0,
        parseFloat(row.Post_frequency) / 10.0
      ]);

      const labels = cleanData.map(row => 
        row.Personality === 'Introvert' ? 1 : 0
      );

      // Split data into training and validation sets
      const splitIndex = Math.floor(features.length * 0.8);
      const trainFeatures = features.slice(0, splitIndex);
      const trainLabels = labels.slice(0, splitIndex);
      const valFeatures = features.slice(splitIndex);
      const valLabels = labels.slice(splitIndex);

      const trainModel = async () => {
        const model = tf.sequential();
        
        model.add(tf.layers.dense({
          inputShape: [7],
          units: 10,
          activation: 'relu'
        }));
        
        model.add(tf.layers.dense({
          units: 5,
          activation: 'relu'
        }));
        
        model.add(tf.layers.dense({
          units: 1,
          activation: 'sigmoid'
        }));

        model.compile({
          optimizer: tf.train.adam(0.001),
          loss: 'binaryCrossentropy',
          metrics: ['accuracy']
        });

        const xs = tf.tensor2d(trainFeatures);
        const ys = tf.tensor2d(trainLabels, [trainLabels.length, 1]);
        const valXs = tf.tensor2d(valFeatures);
        const valYs = tf.tensor2d(valLabels, [valLabels.length, 1]);

        const history = await model.fit(xs, ys, {
          epochs: 50,
          validationData: [valXs, valYs],
          callbacks: {
            onEpochEnd: (epoch, logs) => {
              setTrainingHistory(prev => [...prev, {
                epoch,
                loss: logs.loss,
                accuracy: logs.acc,
                valLoss: logs.val_loss,
                valAccuracy: logs.val_acc
              }]);
            }
          }
        });

        // Calculate metrics on validation set
        const predictions = model.predict(valXs);
        const predArray = await predictions.array();
        const thresholdedPreds = predArray.map(p => p[0] > 0.5 ? 1 : 0);
        
        let tp = 0, fp = 0, tn = 0, fn = 0;
        thresholdedPreds.forEach((pred, i) => {
          if (pred === 1 && valLabels[i] === 1) tp++;
          if (pred === 1 && valLabels[i] === 0) fp++;
          if (pred === 0 && valLabels[i] === 0) tn++;
          if (pred === 0 && valLabels[i] === 1) fn++;
        });

        const accuracy = (tp + tn) / (tp + tn + fp + fn);
        const precision = tp / (tp + fp);
        const recall = tp / (tp + fn);
        const f1Score = 2 * (precision * recall) / (precision + recall);

        // Get the final validation loss from the history object
        const finalValLoss = history.history.val_loss[history.history.val_loss.length - 1];

        setMetrics({
          accuracy,
          precision,
          recall,
          f1Score,
          validationLoss: finalValLoss
        });

        setModel(model);
      };

      await trainModel();
    };

    loadAndPrepareData();
  }, []);

  useEffect(() => {
    if (model) {
      const normalizedInputs = [
        inputs.timeAlone / 11.0,
        inputs.stageFear === 'Yes' ? 1 : 0,
        inputs.socialEvents / 10.0,
        inputs.goingOutside / 7.0,
        inputs.drainedAfterSocializing === 'Yes' ? 1 : 0,
        inputs.friendsCircle / 15.0,
        inputs.postFrequency / 10.0
      ];

      const inputTensor = tf.tensor2d([normalizedInputs]);
      const prediction = model.predict(inputTensor);
      const predictionValue = prediction.dataSync()[0];
      setPrediction(predictionValue);

      const getActivations = async () => {
        const intermediateModel = tf.model({
          inputs: model.input,
          outputs: model.layers.map(layer => layer.output)
        });
        
        const activations = await intermediateModel.predict(inputTensor);
        const activationValues = Array.isArray(activations) 
          ? activations.map(t => t.dataSync())
          : [activations.dataSync()];
        
        setLayerActivations([normalizedInputs, ...activationValues]);
      };
      
      getActivations();
    }
  }, [model, inputs]);

  const chartData = {
    labels: trainingHistory.map((_, index) => `${index + 1}`),
    datasets: [
      {
        label: 'Training Loss',
        data: trainingHistory.map(h => h.loss),
        borderColor: 'rgb(255, 99, 132)',
        tension: 0.1
      },
      {
        label: 'Validation Loss',
        data: trainingHistory.map(h => h.valLoss),
        borderColor: 'rgb(54, 162, 235)',
        tension: 0.1
      },
      {
        label: 'Training Accuracy',
        data: trainingHistory.map(h => h.accuracy),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      },
      {
        label: 'Validation Accuracy',
        data: trainingHistory.map(h => h.valAccuracy),
        borderColor: 'rgb(153, 102, 255)',
        tension: 0.1
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Training Progress'
      }
    },
    scales: {
      y: {
        min: 0,
        max: 1
      }
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        <Paper elevation={3} className="p-6">
          <Typography variant="h4" gutterBottom>
            Personality Prediction Model
          </Typography>
          
          <Box className="mb-6">
            <Typography gutterBottom>Time Spent Alone (hours/day)</Typography>
            <Slider
              value={inputs.timeAlone}
              onChange={(e, value) => setInputs(prev => ({ ...prev, timeAlone: value }))}
              min={0}
              max={11}
              valueLabelDisplay="auto"
            />

            <Typography gutterBottom>Stage Fear</Typography>
            <Button
              variant={inputs.stageFear === 'Yes' ? 'contained' : 'outlined'}
              onClick={() => setInputs(prev => ({ ...prev, stageFear: prev.stageFear === 'Yes' ? 'No' : 'Yes' }))}
              className="mb-4"
            >
              {inputs.stageFear}
            </Button>

            <Typography gutterBottom>Social Event Attendance (per month)</Typography>
            <Slider
              value={inputs.socialEvents}
              onChange={(e, value) => setInputs(prev => ({ ...prev, socialEvents: value }))}
              min={0}
              max={10}
              valueLabelDisplay="auto"
            />

            <Typography gutterBottom>Going Outside (days/week)</Typography>
            <Slider
              value={inputs.goingOutside}
              onChange={(e, value) => setInputs(prev => ({ ...prev, goingOutside: value }))}
              min={0}
              max={7}
              valueLabelDisplay="auto"
            />

            <Typography gutterBottom>Drained After Socializing</Typography>
            <Button
              variant={inputs.drainedAfterSocializing === 'Yes' ? 'contained' : 'outlined'}
              onClick={() => setInputs(prev => ({ 
                ...prev, 
                drainedAfterSocializing: prev.drainedAfterSocializing === 'Yes' ? 'No' : 'Yes' 
              }))}
              className="mb-4"
            >
              {inputs.drainedAfterSocializing}
            </Button>

            <Typography gutterBottom>Friends Circle Size</Typography>
            <Slider
              value={inputs.friendsCircle}
              onChange={(e, value) => setInputs(prev => ({ ...prev, friendsCircle: value }))}
              min={0}
              max={15}
              valueLabelDisplay="auto"
            />

            <Typography gutterBottom>Social Media Post Frequency (per week)</Typography>
            <Slider
              value={inputs.postFrequency}
              onChange={(e, value) => setInputs(prev => ({ ...prev, postFrequency: value }))}
              min={0}
              max={10}
              valueLabelDisplay="auto"
            />
          </Box>

          <Paper elevation={6} className="p-4 mb-6 bg-gradient-to-r from-blue-100 to-purple-100">
            <Typography variant="h5" align="center">
              Personality Prediction: {prediction !== null ? (
                prediction > 0.5 ? 
                `Introvert (${(prediction * 100).toFixed(1)}%)` : 
                `Extrovert (${((1 - prediction) * 100).toFixed(1)}%)`
              ) : 'Training...'}
            </Typography>
          </Paper>

          <Paper elevation={3} className="p-4 mb-6">
            <Typography variant="h6" gutterBottom>Model Performance Metrics</Typography>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Typography variant="subtitle1">Accuracy: {(metrics.accuracy * 100).toFixed(2)}%</Typography>
                <Typography variant="subtitle1">Precision: {(metrics.precision * 100).toFixed(2)}%</Typography>
              </div>
              <div>
                <Typography variant="subtitle1">Recall: {(metrics.recall * 100).toFixed(2)}%</Typography>
                <Typography variant="subtitle1">F1 Score: {(metrics.f1Score * 100).toFixed(2)}%</Typography>
              </div>
            </div>
          </Paper>
        </Paper>

        <Paper elevation={3} className="p-6">
          <Typography variant="h4" gutterBottom>
            Neural Network Visualization
          </Typography>
          <NeuralNetworkViz
            inputs={layerActivations[0] || []}
            weights={[]}
            activations={layerActivations}
          />
        </Paper>

        <Paper elevation={3} className="p-6">
          <div className="h-80">
            <Line data={chartData} options={chartOptions} />
          </div>
        </Paper>

        <Paper elevation={3} className="p-6">
          <Typography variant="h4" gutterBottom>
            Understanding the Personality Prediction
          </Typography>
          
          <Box className="space-y-4">
            <div>
              <Typography variant="h6" gutterBottom>Input Variables and Their Impact</Typography>
              <Typography variant="body1">
                • Time Spent Alone: Higher values suggest introversion
                - Preference for solitude
                - Need for personal space
                - Time for self-reflection
              </Typography>
              
              <Typography variant="body1" className="mt-2">
                • Stage Fear: Strong indicator
                - Common among introverts
                - Related to social anxiety
                - Public speaking comfort level
              </Typography>
              
              <Typography variant="body1" className="mt-2">
                • Social Event Attendance: Key behavioral marker
                - Frequency of social gatherings
                - Comfort in group settings
                - Social energy expenditure
              </Typography>
              
              <Typography variant="body1" className="mt-2">
                • Going Outside: Activity pattern indicator
                - Daily social exposure
                - Comfort with external environment
                - Social interaction frequency
              </Typography>

              <Typography variant="body1" className="mt-2">
                • Drained After Socializing: Energy pattern
                - Social battery indicator
                - Recovery needs
                - Social energy management
              </Typography>

              <Typography variant="body1" className="mt-2">
                • Friends Circle Size: Social network indicator
                - Quality vs quantity of relationships
                - Social comfort zone
                - Relationship maintenance
              </Typography>

              <Typography variant="body1" className="mt-2">
                • Post Frequency: Digital social behavior
                - Online presence
                - Social media comfort
                - Public sharing tendency
              </Typography>
            </div>

            <div>
              <Typography variant="h6" gutterBottom>Neural Network Analysis</Typography>
              <Typography variant="body1">
                The network processes these inputs through:
                1. Input Layer: Normalizes and weighs raw features
                2. Hidden Layers: Identifies complex personality patterns
                3. Output Layer: Generates personality prediction
                4. Continuous Learning: Adapts to new data patterns
              </Typography>
            </div>

            <div>
              <Typography variant="h6" gutterBottom>Current Prediction Analysis</Typography>
              <Typography variant="body1">
                The current prediction is based on the combination of all input factors, with special attention to:
                - Pattern recognition in social behaviors
                - Energy management tendencies
                - Social comfort indicators
                - Communication preferences
              </Typography>
            </div>
          </Box>
        </Paper>
      </div>
    </div>
  );
}

export default App;
