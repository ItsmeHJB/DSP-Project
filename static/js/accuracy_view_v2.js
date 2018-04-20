var data = [{
  samples: 0,
  accuracy: 0
}];

var user_line_data = [{
  samples: 0,
  user_relabelled: 0
}]

var line1_data = [{
  samples: 0,
  accuracy: 0
}];

var line2_data = [{
  samples: 0,
  accuracy: 0
}];

var line3_data = [{
  samples: 0,
  accuracy: 0
}];

var line4_data = [{
  samples: 0,
  accuracy: 0
}];

draw_accuracy_line = true;
draw_user_line = true;


line_plot_colours = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854'];
/*
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
    height = 250 - margin.top - margin.bottom;

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

    if (draw_accuracy_line) {
      g.append("path")
        .datum(data)
        .attr("class", "line")
        .attr("fill", "none")
        .attr("stroke", "steelblue")
        .attr("stroke-linejoin", "round")
        .attr("stroke-linecap", "round")
        .attr("stroke-width", 2)
        .attr("d", line);

    }

    if (draw_user_line) {
      g.append("path")
        .datum(user_label_data)
        .attr("class", "line2")
        .attr("fill", "none")
        .attr("stroke", "red")
        .attr("stroke-linejoin", "round")
        .attr("stroke-linecap", "round")
        .attr("stroke-width", 2)
        .attr("d", line2);
    }


}*/

function create_line_graph_for_all_methods(placement) {
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
    height = 250 - margin.top - margin.bottom;

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
    //.curve(d3.curveMonotoneX)

  var user_line = d3.line()
    .x(function(d) {
      return x(d.samples);
    })
    .y(function(d) {
      return y(d.user_relabelled);
    });
    //.curve(d3.curveMonotoneX)

  //Scale the range of the data
  x.domain(d3.extent(line1_data, function(d) {
    return d.samples;
  }));
  y.domain([0, 100]);

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
    .text("Number of Labelled Samples");

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
        .datum(line1_data)
        .attr("class", "line1")
        .attr("stroke", function() { return line_plot_colours[0] })
        .attr("stroke-width", 2)
        .attr("fill", "none")
        .attr("d", line)
        .style("opacity", 0.8);

    g.append("path")
        .datum(line2_data)
        .attr("class", "line2")
        .attr("stroke", function() { return line_plot_colours[1] })
        .attr("stroke-width", 2)
        .attr("fill", "none")
        .attr("d", line)
        .style("opacity", 0.8);

    g.append("path")
        .datum(line3_data)
        .attr("class", "line3")
        .attr("stroke", function() { return line_plot_colours[2] })
        .attr("stroke-width", 2)
        .attr("fill", "none")
        .attr("d", line)
        .style("opacity", 0.8);

    g.append("path")
        .datum(line4_data)
        .attr("class", "line4")
        .attr("fill", "none")
        .attr("stroke", function() { return line_plot_colours[3] })
        .attr("stroke-width", 2)
        .attr("fill", "none")
        .attr("d", line)
        .style("opacity", 0.8);

    if (draw_user_line) {
      g.append("path")
        .datum(user_line_data)
        .attr("class", "user_line")
        .attr("fill", "none")
        .attr("stroke", function() { return line_plot_colours[4] })
        .attr("stroke-width", 2)
        .attr("fill", "none")
        .attr("d", user_line)
        .style("stroke-dasharray", ("3, 3"))
        .style("opacity", 0.8);
    }

}

function update_line_graph_with_all_methods(incoming_data) {

  console.log("update_line_graph with value ", incoming_data)

  var margin = {
      top: 25,
      right: 50,
      bottom: 50,
      left: 30
    },
    width = 425 - margin.left - margin.right,
    height = 250 - margin.top - margin.bottom;

  line1_data.push({
    samples: incoming_data['samples'],
    accuracy: incoming_data['accuracies']['single_instance']
  });

  line2_data.push({
    samples: incoming_data['samples'],
    accuracy: incoming_data['accuracies']['inferred_label']
  });

  line3_data.push({
    samples: incoming_data['samples'],
    accuracy: incoming_data['accuracies']['data_augmentation']
  });

  line4_data.push({
    samples: incoming_data['samples'],
    accuracy: incoming_data['accuracies']['confidence_augmentation']
  });

  user_line_data.push({
    samples: incoming_data['samples'],
    user_relabelled: incoming_data['accuracies']['user_label_counter']
  });

  var x = d3.scaleLinear()
    .rangeRound([0, width]);

  var y = d3.scaleLinear()
    .rangeRound([height, 0]);

  // Scale the range of the data again
  x.domain(d3.extent(line1_data, function(d) {
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
    });
    //.curve(d3.curveMonotoneX)

  var user_line = d3.line()
    .x(function(d) {
      return x(d.samples);
    })
    .y(function(d) {
      return y(d.user_relabelled);
    });
    //.curve(d3.curveMonotoneX)

  vis.select(".line1") // change the line
    .duration(200)
    .attr("class", "line1")
    .attr("stroke", function() { return line_plot_colours[0] })
    .attr("stroke-width", 2)
    .attr("fill", "none")
    .attr("d", line)
    .style("opacity", 0.8);
  

  vis.select(".line2") // change the line
    .duration(200)
    .attr("class", "line2")
    .attr("stroke", function() { return line_plot_colours[1] })
    .attr("stroke-width", 2)
    .attr("fill", "none")
    .attr("d", line)
    .style("opacity", 0.8);
  
  vis.select(".line3") // change the line
    .duration(200)
    .attr("class", "line3")
    .attr("stroke", function() { return line_plot_colours[2] })
    .attr("stroke-width", 2)
    .attr("fill", "none")
    .attr("d", line)
    .style("opacity", 0.8);
  
  vis.select(".line4") // change the line
    .duration(200)
    .attr("class", "line4")
    .attr("stroke", function() { return line_plot_colours[3] })
    .attr("stroke-width", 2)
    .attr("fill", "none")
    .attr("d", line)
    .style("opacity", 0.8);

  if (draw_user_line) {
      vis.select(".user_line") // change the line
        .duration(200)
        .attr("class", "user_line")
        .attr("stroke", function() { return line_plot_colours[4] })
        .attr("stroke-width", 2)
        .attr("fill", "none")
        .attr("d", user_line)
        .style("opacity", 0.8)
        .style("stroke-dasharray", ("3, 3"));
    }
  

  vis.select(".x.axis") // change the x axis
    .duration(200)
    .call(d3.axisBottom(x));

}

function update_line_graph_with_all_methods_from_file(incoming_data) {

  console.log("update_line_graph with value ", incoming_data)

  var margin = {
      top: 25,
      right: 50,
      bottom: 50,
      left: 30
    },
    width = 425 - margin.left - margin.right,
    height = 250 - margin.top - margin.bottom;

  line1_data.push({
    samples: incoming_data['samples'],
    accuracy: incoming_data['accuracies']['single_instance']
  });

  line2_data.push({
    samples: incoming_data['samples'],
    accuracy: incoming_data['accuracies']['inferred_label']
  });

  line3_data.push({
    samples: incoming_data['samples'],
    accuracy: incoming_data['accuracies']['data_augmentation']
  });

  line4_data.push({
    samples: incoming_data['samples'],
    accuracy: incoming_data['accuracies']['confidence_augmentation']
  });

  user_line_data.push({
    samples: incoming_data['samples'],
    user_relabelled: incoming_data['accuracies']['user_label_counter'] * 10
  });

  var x = d3.scaleLinear()
    .rangeRound([0, width]);

  var y = d3.scaleLinear()
    .rangeRound([height, 0]);

  // Scale the range of the data again
  x.domain(d3.extent(line1_data, function(d) {
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
    });
    //.curve(d3.curveMonotoneX)

  var user_line = d3.line()
    .x(function(d) {
      return x(d.samples);
    })
    .y(function(d) {
      return y(d.user_relabelled);
    });
    //.curve(d3.curveMonotoneX)

  vis.select(".line1") // change the line
    .duration(200)
    .attr("class", "line1")
    .attr("stroke", function() { return line_plot_colours[0] })
    .attr("stroke-width", 2)
    .attr("fill", "none")
    .attr("d", line)
    .style("opacity", 0.8);
  

  vis.select(".line2") // change the line
    .duration(200)
    .attr("class", "line2")
    .attr("stroke", function() { return line_plot_colours[1] })
    .attr("stroke-width", 2)
    .attr("fill", "none")
    .attr("d", line)
    .style("opacity", 0.8);
  
  vis.select(".line3") // change the line
    .duration(200)
    .attr("class", "line3")
    .attr("stroke", function() { return line_plot_colours[2] })
    .attr("stroke-width", 2)
    .attr("fill", "none")
    .attr("d", line)
    .style("opacity", 0.8);
  
  vis.select(".line4") // change the line
    .duration(200)
    .attr("class", "line4")
    .attr("stroke", function() { return line_plot_colours[3] })
    .attr("stroke-width", 2)
    .attr("fill", "none")
    .attr("d", line)
    .style("opacity", 0.8);

  if (draw_user_line) {
      vis.select(".user_line") // change the line
        .duration(200)
        .attr("class", "user_line")
        .attr("stroke", function() { return line_plot_colours[4] })
        .attr("stroke-width", 2)
        .attr("fill", "none")
        .attr("d", user_line)
        .style("opacity", 0.8)
        .style("stroke-dasharray", ("3, 3"));
    }
  

  vis.select(".x.axis") // change the x axis
    .duration(200)
    .call(d3.axisBottom(x));

}


/*
function update_line_graph(line_data, user_label_count) {

  console.log("update_line_graph with value ", user_label_count)

  var margin = {
      top: 25,
      right: 50,
      bottom: 50,
      left: 30
    },
    width = 425 - margin.left - margin.right,
    height = 250 - margin.top - margin.bottom;

  // Append new data to global list
  if (draw_accuracy_line) {
    data.push({
      samples: line_data['samples'],
      accuracy: line_data['accuracy']
    });
  }

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

    if (draw_accuracy_line) {
  // Make the changes
  vis.select(".line") // change the line
    .duration(200)
    .attr("class", "line")
    .attr("fill", "none")
    .attr("stroke", "steelblue")
    .attr("stroke-linejoin", "round")
    .attr("stroke-linecap", "round")
    .attr("stroke-width", 3)
    .attr("d", line);
  }


  if (draw_user_line) {
  vis.select(".line2") // change the line
    .duration(200)
    .attr("class", "line2")
    .attr("fill", "none")
    .attr("stroke", "red")
    .attr("stroke-linejoin", "round")
    .attr("stroke-linecap", "round")
    .attr("stroke-width", 3)
    .attr("d", line2);
  }

  vis.select(".x.axis") // change the x axis
    .duration(200)
    .call(d3.axisBottom(x));

}*/
