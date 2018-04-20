d3.selection.prototype.moveToFront = function() {  
      return this.each(function(){
        this.parentNode.appendChild(this);
      });
    };
    d3.selection.prototype.moveToBack = function() {  
        return this.each(function() { 
            var firstChild = this.parentNode.firstChild; 
            if (firstChild) { 
                this.parentNode.insertBefore(this, firstChild); 
            } 
        });
    };



function create_scatter_plot(data, placement) {

  /*var data = [
    [5, 3],
    [10, 17],
    [8, 4],
    [2, 8],
    [4, 3],
    [16, 17],
    [18, 2],
    [4, 9]
  ];*/

  var margin = {
      top: 25,
      right: 50,
      bottom: 50,
      left: 50
    },
    width = 400 - margin.left - margin.right,
    height = 215 - margin.top - margin.bottom;

  var x = d3.scaleLinear()
    .domain([d3.min(data, function(d) {
      return d[0];
    }), d3.max(data, function(d) {
      return d[0];
    })])
    .range([0, width]);

  var y = d3.scaleLinear()
    .domain([d3.min(data, function(d) {
      return d[1];
    }), d3.max(data, function(d) {
      return d[1];
    })])
    .range([height, 0]);

  var chart = d3.select(placement)
    .append('svg')
    .attr('width', width + margin.right + margin.left)
    .attr('height', height + margin.top + margin.bottom)
    .attr('class', 'chart')

  var main = chart.append('g')
    .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')
    .attr('width', width)
    .attr('height', height)
    .attr('class', 'main')

  // draw the x axis
  var xAxis = d3.axisBottom(x)

  main.append('g')
    .attr('transform', 'translate(0,' + height + ')')
    .attr('class', 'x-axis')
    .call(xAxis);

  // draw the y axis
  var yAxis = d3.axisLeft(y)

  main.append('g')
    .attr('transform', 'translate(0,0)')
    .attr('class', 'y-axis ')
    .call(yAxis);

  var g = main.append("svg:g").attr("id", "points");

  g.selectAll(".dot")
    .data(data)
    .enter().append("circle")
    .attr("id", function(d,i) { return "circle_" + i; })
    .attr("cx", function(d, i) {
      return x(d[0])
    })
    .attr("cy", function(d) {
      return y(d[1])
    })
    .attr("r", 2)
    .style("opacity", 0.1)
    //.style("stroke", "black")
    .style("fill", "grey");
    //.on("mouseover", function(d,i) { d3.select(this).moveToFront(); console.log(d,i); d3.select(this).style("stroke","orange"); })
    //.on("mouseout", function(d,i) { console.log(d,i); d3.select(this).style("stroke","grey"); })
}

function create_scatter_plot_with_all_labels(data, placement) {

  /*var data = [
    [5, 3],
    [10, 17],
    [8, 4],
    [2, 8],
    [4, 3],
    [16, 17],
    [18, 2],
    [4, 9]
  ];*/

  var margin = {
      top: 25,
      right: 50,
      bottom: 50,
      left: 50
    },
    width = 400 - margin.left - margin.right,
    height = 215 - margin.top - margin.bottom;

  var x = d3.scaleLinear()
    .domain([d3.min(data, function(d) {
      return d[0];
    }), d3.max(data, function(d) {
      return d[0];
    })])
    .range([0, width]);

  var y = d3.scaleLinear()
    .domain([d3.min(data, function(d) {
      return d[1];
    }), d3.max(data, function(d) {
      return d[1];
    })])
    .range([height, 0]);

  var chart = d3.select(placement)
    .append('svg')
    .attr('width', width + margin.right + margin.left)
    .attr('height', height + margin.top + margin.bottom)
    .attr('class', 'chart')

  var main = chart.append('g')
    .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')
    .attr('width', width)
    .attr('height', height)
    .attr('class', 'main')

  // draw the x axis
  var xAxis = d3.axisBottom(x)

  main.append('g')
    .attr('transform', 'translate(0,' + height + ')')
    .attr('class', 'x-axis')
    .call(xAxis);

  // draw the y axis
  var yAxis = d3.axisLeft(y)

  main.append('g')
    .attr('transform', 'translate(0,0)')
    .attr('class', 'y-axis ')
    .call(yAxis);

  var g = main.append("svg:g").attr("id", "points");

  g.selectAll(".dot")
    .data(data)
    .enter().append("circle")
    .attr("id", function(d,i) { return "circle_" + i; })
    .attr("cx", function(d, i) {
      return x(d[0])
    })
    .attr("cy", function(d) {
      return y(d[1])
    })
    .attr("r", 2)
    .style("opacity", 0.1)
    //.style("stroke", "black")
    .style("fill", function(d) {return our_ten_colours[d[2]] })
    //.on("mouseover", function(d,i) { d3.select(this).moveToFront(); console.log(d,i); d3.select(this).style("stroke","orange"); })
    //.on("mouseout", function(d,i) { console.log(d,i); d3.select(this).style("stroke","grey"); })
}

function highlight_point_in_scatterplot(image_id, label) {
  console.log("does this work?", image_id)
  d3.select("#scatter_plot-view")
    .select("svg")
    .select("#circle_" + image_id)
    .style("opacity", 1)
    .style("fill", function() {return our_ten_colours[label] })
        .moveToFront();
}
