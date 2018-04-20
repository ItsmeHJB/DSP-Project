var data = [{
  samples: 0,
  accuracy: 0
}];

var user_label_data = [{
  samples: 0,
  user_label: 0
}]

function create_line_graph(line_data, placement) {
  // https://bl.ocks.org/mbostock/3883245
  // http://bl.ocks.org/d3noob/b3ff6ae1c120eea654b5
  // http://jsfiddle.net/spanndemic/tyLvshfa/

  var margin = {
      top: 25,
      right: 50,
      bottom: 50,
      left: 30
    },
    width = 425 - margin.left - margin.right,
    height = 380 - margin.top - margin.bottom;

  var vis = d3.select(placement)
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)

  var g = vis.append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  var tooltip = d3.select(placement).append('div')
    .style('postion', 'absolute')
    .style('padding', '0 10px')
    .style('background', 'white')
    .style('opacity', 0)

  var x = d3.scaleLinear()
    .rangeRound([0, width]);

  var y = d3.scaleLinear()
    .rangeRound([height, 0]);

  var line = d3.line()
    .x(function(d) {
      return x(d.samples);
    })
    .y(function(d) {
      return y(d.accuracy);
    })
    .curve(d3.curveMonotoneX)

  var line2 = d3.line()
    .x(function(d) {
      return x(d.samples);
    })
    .y(function(d) {
      return y(d.user_labels);
    })
    .curve(d3.curveMonotoneX)

  //Scale the range of the data
  x.domain(d3.extent(data, function(d) {
    return d.samples;
  }));
  y.domain([0, 100]);
  // y.domain([0, d3.max(data, function(d) { return d.accuracy; })]);


  g.append("g")
    .attr("transform", "translate(0," + height + ")")
    .attr("class", "x axis")
    .call(d3.axisBottom(x))
    .append("text")
    .attr("fill", "#000")
    .attr("x", width / 2)
    .attr("y", 22)
    .attr("dy", "0.71em")
    .attr("text-anchor", "middle")
    .text("Number of Samples Trained");
  // .select(".domain")
  //   .remove();

  g.append("g")
    .attr("class", "y axis")
    .call(d3.axisLeft(y))
    .append("text")
    .attr("fill", "#000")
    .attr("transform", "rotate(-90)")
    .attr("y", 6)
    .attr("dy", "0.71em")
    .attr("text-anchor", "end");
    //.text("Accuracy (%)");


  g.append("path")
    .datum(data)
    .attr("class", "line")
    .attr("fill", "none")
    .attr("stroke", "steelblue")
    .attr("stroke-linejoin", "round")
    .attr("stroke-linecap", "round")
    .attr("stroke-width", 2)
    .attr("d", line);

  g.append("path")
    .datum(user_label_data)
    .attr("class", "line2")
    .attr("fill", "none")
    .attr("stroke", "red")
    .attr("stroke-linejoin", "round")
    .attr("stroke-linecap", "round")
    .attr("stroke-width", 2)
    .attr("d", line2);


/*
  g.selectAll(".dot")
    .data(data)
    .enter().append("circle")
    .attr("class", "dot")
    .attr("cx", function(d) {
      return x(d.samples)
    })
    .attr("cy", function(d) {
      return y(d.accuracy)
    })
    .attr("r", 5)

    .on('mouseover', function(d) {
      tooltip.transition()

      tempColour = this.style.fill;
      d3.select(this)
        .style('fill', 'red')
    })
    .on('mouseout', function(d) {
      d3.select(this)
        .style('fill', tempColour)
    })
*/

}

function update_line_graph(line_data, user_label_count) {

  console.log("update_line_graph with value ", user_label_count)

  var margin = {
      top: 25,
      right: 50,
      bottom: 50,
      left: 30
    },
    width = 400 - margin.left - margin.right,
    height = 380 - margin.top - margin.bottom;

  // Append new data to global list
  data.push({
    samples: line_data['samples'],
    accuracy: line_data['accuracy']
  });

// originally we just showed the value as it was
  //user_label_data.push({
  //  samples: line_data['samples'],
  //  user_label: user_label_count
  //})

// we now show this as a percentage of the full training set (i.e., X / 55000)
// on the basis that if we had 55000 correct interactions we would have a fully labelled training set
  user_label_data.push({
    samples: line_data['samples'],
    //user_label: parseFloat(user_label_count / 55000)
    user_label: parseFloat(user_label_count)
  })

  var x = d3.scaleLinear()
    .rangeRound([0, width]);

  var y = d3.scaleLinear()
    .rangeRound([height, 0]);

  // Scale the range of the data again
  x.domain(d3.extent(data, function(d) {
    return d.samples;
  }));
  y.domain([0, 100]);
  // y.domain([0, d3.max(data, function(d) { return d.accuracy; })]);

  // Select the section we want to apply our changes to
  var vis = d3.select('#line_graph-view').transition();

  var line = d3.line()
    .x(function(d) {
      return x(d.samples);
    })
    .y(function(d) {
      return y(d.accuracy);
    })
  // .curve(d3.curveMonotoneX)

  var line2 = d3.line()
    .x(function(d) {
      return x(d.samples);
    })
    .y(function(d) {
      return y(d.user_label);
    })
    //.curve(d3.curveMonotoneX)

  // Make the changes
  vis.select(".line") // change the line
    .duration(750)
    .attr("class", "line")
    .attr("fill", "none")
    .attr("stroke", "steelblue")
    .attr("stroke-linejoin", "round")
    .attr("stroke-linecap", "round")
    .attr("stroke-width", 3)
    .attr("d", line);

  vis.select(".line2") // change the line
    .duration(750)
    .attr("class", "line2")
    .attr("fill", "none")
    .attr("stroke", "red")
    .attr("stroke-linejoin", "round")
    .attr("stroke-linecap", "round")
    .attr("stroke-width", 3)
    .attr("d", line2);

  vis.select(".x.axis") // change the x axis
    .duration(750)
    .call(d3.axisBottom(x));

  // vis.selectAll(".dot")
  //   .duration(750)
  //   .data(data)
  //   .enter().append("circle")
  //   .attr("class", "dot")
  //   .attr("cx", function(d) {
  //     return x(d.samples)
  //   })
  //   .attr("cy", function(d) {
  //     return y(d.accuracy)
  //   })
  //   .attr("r", 5)
  //
  //   .on('mouseover', function(d) {
  //     tooltip.transition()
  //
  //     tempColour = this.style.fill;
  //     d3.select(this)
  //       .style('fill', 'red')
  //   })
  //   .on('mouseout', function(d) {
  //     d3.select(this)
  //       .style('fill', tempColour)
  //   })

}
