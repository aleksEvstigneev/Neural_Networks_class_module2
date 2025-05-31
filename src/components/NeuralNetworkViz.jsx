import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { motion } from 'framer-motion';

const NeuralNetworkViz = ({ inputs, weights, activations }) => {
  const svgRef = useRef();
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  const layers = [
    { nodes: 7, name: 'Input Layer' },
    { nodes: 10, name: 'Hidden Layer 1' },
    { nodes: 5, name: 'Hidden Layer 2' },
    { nodes: 1, name: 'Output Layer' }
  ];

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 40, right: 40, bottom: 40, left: 40 };
    const width = dimensions.width - margin.left - margin.right;
    const height = dimensions.height - margin.top - margin.bottom;

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Add layer labels
    g.selectAll(".layer-label")
      .data(layers)
      .enter()
      .append("text")
      .attr("class", "layer-label")
      .attr("x", (d, i) => (width * i) / (layers.length - 1))
      .attr("y", -20)
      .attr("text-anchor", "middle")
      .style("font-size", "14px")
      .style("font-weight", "bold")
      .text(d => d.name);

    // Calculate node positions
    const layerSpacing = width / (layers.length - 1);
    const nodes = [];
    const links = [];

    layers.forEach((layer, i) => {
      const nodeSpacing = height / (layer.nodes + 1);
      for (let j = 0; j < layer.nodes; j++) {
        nodes.push({
          x: i * layerSpacing,
          y: (j + 1) * nodeSpacing,
          layer: i,
          index: j,
          value: activations?.[i]?.[j] || 0
        });
      }
    });

    // Create connections between layers
    for (let i = 0; i < layers.length - 1; i++) {
      const currentLayer = nodes.filter(n => n.layer === i);
      const nextLayer = nodes.filter(n => n.layer === i + 1);
      
      currentLayer.forEach(current => {
        nextLayer.forEach(next => {
          links.push({
            source: current,
            target: next,
            weight: weights?.[i]?.[current.index]?.[next.index] || Math.random() * 2 - 1
          });
        });
      });
    }

    // Draw connections with gradient
    const linkGroup = g.append("g").attr("class", "links");
    
    links.forEach(link => {
      const gradientId = `gradient-${link.source.layer}-${link.source.index}-${link.target.index}`;
      
      // Create gradient
      const gradient = svg.append("defs")
        .append("linearGradient")
        .attr("id", gradientId)
        .attr("x1", "0%")
        .attr("y1", "0%")
        .attr("x2", "100%")
        .attr("y2", "0%");

      gradient.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", link.weight > 0 ? "#4CAF50" : "#f44336")
        .attr("stop-opacity", Math.abs(link.weight));

      gradient.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", link.weight > 0 ? "#4CAF50" : "#f44336")
        .attr("stop-opacity", Math.abs(link.weight));

      // Draw connection
      linkGroup.append("path")
        .attr("d", `M${link.source.x},${link.source.y} C${(link.source.x + link.target.x) / 2},${link.source.y} ${(link.source.x + link.target.x) / 2},${link.target.y} ${link.target.x},${link.target.y}`)
        .style("fill", "none")
        .style("stroke", `url(#${gradientId})`)
        .style("stroke-width", Math.abs(link.weight) * 2);
    });

    // Draw nodes
    const nodeGroup = g.append("g").attr("class", "nodes");
    
    const node = nodeGroup.selectAll(".node")
      .data(nodes)
      .enter()
      .append("g")
      .attr("class", "node")
      .attr("transform", d => `translate(${d.x},${d.y})`);

    // Node circles
    node.append("circle")
      .attr("r", 20)
      .style("fill", d => d3.interpolateViridis(d.value))
      .style("stroke", "#666")
      .style("stroke-width", 2);

    // Node values
    node.append("text")
      .attr("dy", ".35em")
      .attr("text-anchor", "middle")
      .style("fill", "white")
      .style("font-size", "12px")
      .text(d => d.value.toFixed(2));

    // Add feature labels for input layer
    const inputLabels = [
      "Time Alone",
      "Stage Fear",
      "Social Events",
      "Going Outside",
      "Drained After",
      "Friends Circle",
      "Post Frequency"
    ];

    nodeGroup.selectAll(".input-label")
      .data(nodes.filter(n => n.layer === 0))
      .enter()
      .append("text")
      .attr("class", "input-label")
      .attr("x", d => d.x - 30)
      .attr("y", d => d.y)
      .attr("text-anchor", "end")
      .attr("dy", ".35em")
      .style("font-size", "12px")
      .text((d, i) => inputLabels[i]);

  }, [dimensions, weights, activations]);

  return (
    <div className="relative w-full h-full">
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        className="border rounded-lg shadow-lg bg-white"
        style={{ maxWidth: '100%', height: 'auto' }}
      />
      <div className="mt-6 p-4 bg-white rounded-lg shadow-lg">
        <h3 className="text-lg font-semibold mb-3">Understanding the Neural Network Visualization</h3>
        <div className="space-y-4">
          <div>
            <h4 className="font-medium">Node Values</h4>
            <p className="text-gray-700">The numbers in each node represent activation values (between 0 and 1):</p>
            <ul className="list-disc ml-6 mt-2">
              <li>Input Layer: Normalized input values for each personality feature</li>
              <li>Hidden Layers: Intermediate features learned by the network</li>
              <li>Output Layer: Final prediction (closer to 1 = Introvert, closer to 0 = Extrovert)</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium">Node Colors</h4>
            <p className="text-gray-700">The color intensity represents the activation strength - darker colors indicate higher values.</p>
          </div>
          <div>
            <h4 className="font-medium">Connections</h4>
            <ul className="list-disc ml-6">
              <li>Green lines: Positive weights (reinforcing connections)</li>
              <li>Red lines: Negative weights (inhibiting connections)</li>
              <li>Line thickness: Magnitude of the weight</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NeuralNetworkViz;
